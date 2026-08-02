export class InferenceEngine {
    constructor() {
        this.session = null;
    }

    async init(version) {
        if (!window.ort) {
            throw new Error('ONNX Runtime did not load. Check your connection and refresh.');
        }
        window.ort.env.wasm.numThreads = 1;

        this.session = await window.ort.InferenceSession.create(`FFN/model_nn.onnx?v=${version}`, {
            executionProviders: ['wasm']
        });

        const inputNames = this.session.inputNames || [];
        if (inputNames.length !== 1 || inputNames[0] !== 'x_deep') {
            this.session = null;
            throw new Error('The prediction model is from an older export. Train and export the current model, then refresh.');
        }

        console.log('[ONNX] Prediction model loaded.');
    }

    async runInferenceBatch(batchDeep) {
        if (!this.session) throw new Error('Prediction model is not ready.');
        if (!Array.isArray(batchDeep) || batchDeep.length === 0) return new Float32Array();

        const numSamples = batchDeep.length;
        const deepFlat = new BigInt64Array(numSamples * 10);

        for (let s = 0; s < numSamples; s++) {
            if (!Array.isArray(batchDeep[s]) || batchDeep[s].length !== 10) {
                throw new Error('Each draft must contain exactly 10 champion slots.');
            }
            for (let j = 0; j < 10; j++) {
                deepFlat[s * 10 + j] = BigInt(batchDeep[s][j]);
            }
        }

        const tDeep = new window.ort.Tensor('int64', deepFlat, [numSamples, 10]);
        const results = await this.session.run({ x_deep: tDeep });
        
        const outProbs = new Float32Array(numSamples);
        const rawOut = results.output || Object.values(results)[0];
        if (!rawOut?.data || rawOut.data.length < numSamples) {
            throw new Error('Prediction model returned an invalid output.');
        }

        for (let s = 0; s < numSamples; s++) {
            const value = Number(rawOut.data[s]);
            if (!Number.isFinite(value)) {
                throw new Error('Prediction model returned a non-numeric probability.');
            }
            outProbs[s] = Math.min(1, Math.max(0, value));
        }
        return outProbs;
    }
}
