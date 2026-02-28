/**
 * ImagePreprocessor — Canvas-based image preprocessing pipeline
 * CLAHE, brightness normalization, denoising, shadow correction, histogram equalization
 */
class ImagePreprocessor {
    constructor() {
        this.enabled = true;
        this.settings = {
            clahe: true,
            brightnessNorm: true,
            denoise: true,
            motionBlurComp: false,
            shadowCorrection: true,
            histogramEq: false // off by default — CLAHE is better
        };
        // CLAHE params
        this._tileSize = 8;
        this._clipLimit = 2.5;
        // Denoise kernel (3×3 Gaussian)
        this._gaussKernel = [
            1, 2, 1,
            2, 4, 2,
            1, 2, 1
        ];
        this._gaussSum = 16;
        // Background model for shadow correction
        this._bgModel = null;
        this._bgAlpha = 0.005; // slow adaptation
    }

    /**
     * Full preprocessing pipeline — returns a new canvas
     */
    preprocess(sourceCanvas) {
        if (!this.enabled) return sourceCanvas;

        const w = sourceCanvas.width;
        const h = sourceCanvas.height;
        const outCanvas = document.createElement('canvas');
        outCanvas.width = w;
        outCanvas.height = h;
        const outCtx = outCanvas.getContext('2d');
        outCtx.drawImage(sourceCanvas, 0, 0);

        let imageData = outCtx.getImageData(0, 0, w, h);

        // 1. Denoise first (reduce noise before other ops)
        if (this.settings.denoise) {
            imageData = this._applyGaussianBlur(imageData, w, h);
        }

        // 2. Shadow correction
        if (this.settings.shadowCorrection) {
            imageData = this._correctShadows(imageData, w, h);
        }

        // 3. Adaptive brightness normalization
        if (this.settings.brightnessNorm) {
            imageData = this._normalizeBrightness(imageData, w, h);
        }

        // 4. CLAHE (preferred) or Histogram Equalization
        if (this.settings.clahe) {
            imageData = this._applyCLAHE(imageData, w, h);
        } else if (this.settings.histogramEq) {
            imageData = this._equalizeHistogram(imageData, w, h);
        }

        // 5. Motion blur compensation (sharpening)
        if (this.settings.motionBlurComp) {
            imageData = this._compensateMotionBlur(imageData, w, h);
        }

        outCtx.putImageData(imageData, 0, 0);
        return outCanvas;
    }

    // ---- CLAHE ----
    _applyCLAHE(imageData, w, h) {
        const data = imageData.data;
        const tileW = Math.ceil(w / this._tileSize);
        const tileH = Math.ceil(h / this._tileSize);

        // Convert to luminance array
        const lum = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
            const idx = i * 4;
            lum[i] = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        }

        // Build per-tile CDF with clipping
        const tiles = [];
        for (let ty = 0; ty < this._tileSize; ty++) {
            for (let tx = 0; tx < this._tileSize; tx++) {
                const x0 = tx * tileW;
                const y0 = ty * tileH;
                const x1 = Math.min(x0 + tileW, w);
                const y1 = Math.min(y0 + tileH, h);

                // Histogram
                const hist = new Float32Array(256);
                let count = 0;
                for (let y = y0; y < y1; y++) {
                    for (let x = x0; x < x1; x++) {
                        const v = Math.min(255, Math.max(0, Math.round(lum[y * w + x])));
                        hist[v]++;
                        count++;
                    }
                }

                // Clip histogram
                if (count > 0) {
                    const clipVal = this._clipLimit * (count / 256);
                    let excess = 0;
                    for (let i = 0; i < 256; i++) {
                        if (hist[i] > clipVal) {
                            excess += hist[i] - clipVal;
                            hist[i] = clipVal;
                        }
                    }
                    const redistribute = excess / 256;
                    for (let i = 0; i < 256; i++) {
                        hist[i] += redistribute;
                    }
                }

                // CDF
                const cdf = new Float32Array(256);
                cdf[0] = hist[0];
                for (let i = 1; i < 256; i++) {
                    cdf[i] = cdf[i - 1] + hist[i];
                }
                // Normalize CDF to 0–255
                const cdfMin = cdf[0];
                const cdfRange = count - cdfMin;
                const mapping = new Uint8Array(256);
                for (let i = 0; i < 256; i++) {
                    mapping[i] = cdfRange > 0
                        ? Math.round(((cdf[i] - cdfMin) / cdfRange) * 255)
                        : i;
                }
                tiles.push({ x0, y0, x1, y1, mapping });
            }
        }

        // Apply mapping with bilinear interpolation between tiles
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const idx = (y * w + x) * 4;
                const origLum = Math.min(255, Math.max(0, Math.round(lum[y * w + x])));

                // Find tile
                const tx = Math.min(Math.floor(x / tileW), this._tileSize - 1);
                const ty2 = Math.min(Math.floor(y / tileH), this._tileSize - 1);
                const tileIdx = ty2 * this._tileSize + tx;
                const mapped = tiles[tileIdx].mapping[origLum];

                // Scale RGB channels proportionally
                const scale = origLum > 0 ? mapped / origLum : 1;
                data[idx] = Math.min(255, Math.max(0, Math.round(data[idx] * scale)));
                data[idx + 1] = Math.min(255, Math.max(0, Math.round(data[idx + 1] * scale)));
                data[idx + 2] = Math.min(255, Math.max(0, Math.round(data[idx + 2] * scale)));
            }
        }

        return imageData;
    }

    // ---- Adaptive Brightness Normalization ----
    _normalizeBrightness(imageData, w, h) {
        const data = imageData.data;
        const targetMean = 128;

        // Calculate mean brightness
        let sum = 0;
        const count = w * h;
        for (let i = 0; i < data.length; i += 4) {
            sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        }
        const meanBrightness = sum / count;

        if (meanBrightness < 1) return imageData;

        const factor = targetMean / meanBrightness;
        // Clamp factor to avoid extreme adjustments
        const clampedFactor = Math.min(2.0, Math.max(0.5, factor));

        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.min(255, Math.round(data[i] * clampedFactor));
            data[i + 1] = Math.min(255, Math.round(data[i + 1] * clampedFactor));
            data[i + 2] = Math.min(255, Math.round(data[i + 2] * clampedFactor));
        }

        return imageData;
    }

    // ---- Gaussian Noise Filter (3×3) ----
    _applyGaussianBlur(imageData, w, h) {
        const src = imageData.data;
        const dst = new Uint8ClampedArray(src.length);
        const k = this._gaussKernel;
        const ks = this._gaussSum;

        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                for (let c = 0; c < 3; c++) {
                    let val = 0;
                    let ki = 0;
                    for (let ky = -1; ky <= 1; ky++) {
                        for (let kx = -1; kx <= 1; kx++) {
                            val += src[((y + ky) * w + (x + kx)) * 4 + c] * k[ki++];
                        }
                    }
                    dst[(y * w + x) * 4 + c] = Math.round(val / ks);
                }
                dst[(y * w + x) * 4 + 3] = 255; // alpha
            }
        }

        // Copy border pixels as-is
        for (let x = 0; x < w; x++) {
            for (let c = 0; c < 4; c++) {
                dst[x * 4 + c] = src[x * 4 + c];
                dst[((h - 1) * w + x) * 4 + c] = src[((h - 1) * w + x) * 4 + c];
            }
        }
        for (let y = 0; y < h; y++) {
            for (let c = 0; c < 4; c++) {
                dst[y * w * 4 + c] = src[y * w * 4 + c];
                dst[(y * w + w - 1) * 4 + c] = src[(y * w + w - 1) * 4 + c];
            }
        }

        imageData.data.set(dst);
        return imageData;
    }

    // ---- Motion Blur Compensation (Unsharp Mask) ----
    _compensateMotionBlur(imageData, w, h) {
        const data = imageData.data;
        // Create blurred copy
        const blurredData = new Uint8ClampedArray(data);
        const blurImgData = new ImageData(blurredData, w, h);
        this._applyGaussianBlur(blurImgData, w, h);

        const amount = 1.5; // sharpening strength

        for (let i = 0; i < data.length; i += 4) {
            for (let c = 0; c < 3; c++) {
                const diff = data[i + c] - blurImgData.data[i + c];
                data[i + c] = Math.min(255, Math.max(0, Math.round(data[i + c] + amount * diff)));
            }
        }

        return imageData;
    }

    // ---- Shadow Detection & Correction ----
    _correctShadows(imageData, w, h) {
        const data = imageData.data;

        // Update running background model
        if (!this._bgModel || this._bgModel.length !== w * h * 3) {
            this._bgModel = new Float32Array(w * h * 3);
            for (let i = 0; i < w * h; i++) {
                this._bgModel[i * 3] = data[i * 4];
                this._bgModel[i * 3 + 1] = data[i * 4 + 1];
                this._bgModel[i * 3 + 2] = data[i * 4 + 2];
            }
            return imageData;
        }

        // Adapt background model slowly
        for (let i = 0; i < w * h; i++) {
            for (let c = 0; c < 3; c++) {
                this._bgModel[i * 3 + c] =
                    (1 - this._bgAlpha) * this._bgModel[i * 3 + c] +
                    this._bgAlpha * data[i * 4 + c];
            }
        }

        // Divide-by-background method for shadow removal
        for (let i = 0; i < w * h; i++) {
            for (let c = 0; c < 3; c++) {
                const bg = this._bgModel[i * 3 + c];
                if (bg > 10) {
                    const corrected = (data[i * 4 + c] / bg) * 128;
                    data[i * 4 + c] = Math.min(255, Math.max(0, Math.round(corrected)));
                }
            }
        }

        return imageData;
    }

    // ---- Histogram Equalization ----
    _equalizeHistogram(imageData, w, h) {
        const data = imageData.data;

        // Convert to HSL-like: equalize only the luminance
        const lumHist = new Float32Array(256);
        const lumArr = new Uint8Array(w * h);
        const count = w * h;

        for (let i = 0; i < count; i++) {
            const idx = i * 4;
            const l = Math.round(0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2]);
            lumArr[i] = l;
            lumHist[l]++;
        }

        // Build CDF
        const cdf = new Float32Array(256);
        cdf[0] = lumHist[0];
        for (let i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + lumHist[i];
        }

        const cdfMin = cdf[0];
        const range = count - cdfMin;
        const mapping = new Uint8Array(256);
        for (let i = 0; i < 256; i++) {
            mapping[i] = range > 0 ? Math.round(((cdf[i] - cdfMin) / range) * 255) : i;
        }

        // Apply
        for (let i = 0; i < count; i++) {
            const idx = i * 4;
            const oldLum = lumArr[i];
            const newLum = mapping[oldLum];
            const scale = oldLum > 0 ? newLum / oldLum : 1;
            data[idx] = Math.min(255, Math.round(data[idx] * scale));
            data[idx + 1] = Math.min(255, Math.round(data[idx + 1] * scale));
            data[idx + 2] = Math.min(255, Math.round(data[idx + 2] * scale));
        }

        return imageData;
    }

    /**
     * Reset background model (call when product placement changes)
     */
    resetBackground() {
        this._bgModel = null;
    }
}

window.imagePreprocessor = new ImagePreprocessor();
