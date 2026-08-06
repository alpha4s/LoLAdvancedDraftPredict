import { InferenceEngine } from './model.js';
import { UIController } from './ui.js';

const ROLES = ['top', 'jungle', 'mid', 'bot', 'support'];

document.addEventListener('DOMContentLoaded', async () => {
    let champs = [];
    let champToIdx = {};
    let numChamps = 0;
    let modelReady = false;
    let requestId = 0;

    const inferenceEngine = new InferenceEngine();
    const ui = new UIController(onDraftChange);

    const version = Date.now();
    const [cData, mData] = await Promise.all([
        fetch(`champions.json?v=${version}`).then(res => res.json()),
        fetch(`FFN/model_nn_metadata.json?v=${version}`).then(res => res.json())
    ]);

    champs = cData;
    champToIdx = mData.champ_to_idx;
    numChamps = champs.length;

    ui.renderGrid(champs);
    await inferenceEngine.init(version);
    modelReady = true;
    onDraftChange();

    async function onDraftChange() {
        if (!modelReady) return;
        const currentRequest = ++requestId;
        const bNames = ROLES.map(r => ui.getSlotName('blue', r));
        const rNames = ROLES.map(r => ui.getSlotName('red', r));

        const bIdxs = bNames.map(c => champToIdx[c] ?? numChamps);
        const rIdxs = rNames.map(c => champToIdx[c] ?? numChamps);

        const baseDeep = [...bIdxs, ...rIdxs];
        const isDefault = bNames.every(c => !c) && rNames.every(c => !c);
        const [baseProb] = isDefault ? [0.5] : await inferenceEngine.runInferenceBatch([baseDeep]);

        if (currentRequest !== requestId) return;
        ui.updateWinRates(baseProb);

        if (!ui.targetCard) return ui.renderRecommendations([]);

        const { team, role } = ui.targetCard.dataset;
        const flatSlotIdx = (team === 'blue' ? 0 : 5) + ROLES.indexOf(role);

        const pool = ui.pool.length ? ui.pool : champs;
        const picked = new Set([...bNames, ...rNames].filter(Boolean));
        const candidates = pool.filter(c => champToIdx[c] !== undefined && !picked.has(c));

        if (!candidates.length) return ui.renderRecommendations([]);

        const batchDeep = candidates.map(cand => {
            const sample = [...baseDeep];
            sample[flatSlotIdx] = champToIdx[cand] ?? numChamps;
            return sample;
        });

        const candProbs = await inferenceEngine.runInferenceBatch(batchDeep);
        if (currentRequest !== requestId) return;

        const recs = candidates.map((name, i) => ({
            name,
            delta: team === 'blue' ? candProbs[i] - baseProb : baseProb - candProbs[i]
        }));

        recs.sort((a, b) => b.delta - a.delta);
        ui.renderRecommendations(recs.slice(0, 10));
    }
});
