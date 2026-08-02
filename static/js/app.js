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
    try {
        const [cData, mData] = await Promise.all([
            fetchJson(`champions.json?v=${version}`),
            fetchJson(`FFN/model_nn_metadata.json?v=${version}`)
        ]);

        champs = Array.isArray(cData) ? cData : Object.values(cData);
        champToIdx = mData.champ_to_idx || {};
        numChamps = Object.keys(champToIdx).length || champs.length;

        ui.renderGrid(champs);
        try {
            await inferenceEngine.init(version);
            modelReady = true;
            ui.setStatus('Prediction model ready', 'ready');
            onDraftChange();
        } catch (modelError) {
            console.error('Model initialization failed:', modelError);
            ui.setStatus(modelError.message || 'Prediction model unavailable.', 'error');
        }
    } catch (err) {
        console.error('Interface initialization failed:', err);
        ui.setStatus('Could not load champion data. Refresh to try again.', 'error');
    }

    async function fetchJson(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load ${url} (${response.status})`);
        return response.json();
    }

    async function onDraftChange() {
        if (!modelReady) return;
        const currentRequest = ++requestId;
        const bNames = ROLES.map(r => ui.getSlotName('blue', r));
        const rNames = ROLES.map(r => ui.getSlotName('red', r));

        const bIdxs = bNames.map(c => champToIdx[c] ?? numChamps);
        const rIdxs = rNames.map(c => champToIdx[c] ?? numChamps);

        const baseDeep = [...bIdxs, ...rIdxs];
        let baseProb;
        try {
            [baseProb] = await inferenceEngine.runInferenceBatch([baseDeep]);
        } catch (error) {
            if (currentRequest === requestId) handleInferenceError(error);
            return;
        }
        if (currentRequest !== requestId) return;
        ui.updateWinRates(baseProb);

        if (!ui.targetCard) {
            ui.renderRecommendations([]);
            return;
        }

        const team = ui.targetCard.dataset.team;
        const role = ui.targetCard.dataset.role;
        const targetIdx = ROLES.indexOf(role);
        const targetSideOffset = team === 'blue' ? 0 : 5;
        const flatSlotIdx = targetSideOffset + targetIdx;

        const pool = ui.pool.length ? ui.pool : champs;
        const picked = new Set([...bNames, ...rNames].filter(Boolean));
        const candidates = pool.filter(c => Object.hasOwn(champToIdx, c) && !picked.has(c));

        if (candidates.length === 0) {
            ui.renderRecommendations([]);
            return;
        }

        const batchDeep = [];
        candidates.forEach(cand => {
            const dSample = [...baseDeep];
            dSample[flatSlotIdx] = champToIdx[cand] ?? numChamps;
            batchDeep.push(dSample);
        });

        let candProbs;
        try {
            candProbs = await inferenceEngine.runInferenceBatch(batchDeep);
        } catch (error) {
            if (currentRequest === requestId) handleInferenceError(error);
            return;
        }
        if (currentRequest !== requestId) return;

        const recs = candidates.map((cand, i) => {
            const prob = candProbs[i];
            const delta = team === 'blue' ? (prob - baseProb) : (baseProb - prob);
            return { name: cand, delta: delta };
        });

        recs.sort((a, b) => b.delta - a.delta);
        ui.renderRecommendations(recs.slice(0, 10));
    }

    function handleInferenceError(error) {
        console.error('Prediction failed:', error);
        modelReady = false;
        requestId++;
        ui.renderRecommendations([]);
        ui.setStatus('Prediction failed. Refresh the page to reload the model.', 'error');
    }
});
