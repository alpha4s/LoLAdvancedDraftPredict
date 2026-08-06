export class InferenceEngine {
    constructor() {
        this.session = null;
    }

    async init(version) {
        window.ort.env.wasm.numThreads = 1;
        this.session = await window.ort.InferenceSession.create(`FFN/model_nn.onnx?v=${version}`, {
            executionProviders: ['wasm']
        });
        console.log('[ONNX] Prediction model loaded.');
    }

    async runInferenceBatch(batchDeep) {
        if (!this.session || !batchDeep.length) return new Float32Array();

        const deepFlat = BigInt64Array.from(batchDeep.flat(), BigInt);
        const tDeep = new window.ort.Tensor('int64', deepFlat, [batchDeep.length, 10]);
        const results = await this.session.run({ x_deep: tDeep });

        return Float32Array.from(results.output.data);
    }
}
