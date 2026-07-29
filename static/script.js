const API = (location.hostname === 'localhost' || location.hostname === '127.0.0.1')
    ? ''
    : 'https://loladvanceddraftpredict.onrender.com';

const post = (url, body) => fetch(API + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
}).then(r => r.json()).catch(() => ({}));

const $ = id => document.getElementById(id);

document.addEventListener('DOMContentLoaded', () => {
    let champs = [];
    let activeSlot = null;
    let targetCard = null;
    let pool = [];
    let dragCard = null;

    try {
        pool = JSON.parse(localStorage.getItem('my_personal_champion_pool')) || [];
    } catch (e) {}

    const grid = $('champion-grid');
    const search = $('pool-search');
    const clearBtn = $('clear-btn');
    const swapBtn = $('swap-teams-btn');
    const cards = document.querySelectorAll('.role-card');
    const recBox = $('recommendations-box');
    const recList = $('recommendations-list');
    const bluePct = $('blue-percent');
    const redPct = $('red-percent');

    fetch('champions.json').then(r => r.json()).then(data => {
        champs = data;
        renderGrid();
        predict();
    });

    const savePool = () => {
        try {
            localStorage.setItem('my_personal_champion_pool', JSON.stringify(pool));
        } catch (e) {}
    };

    function renderGrid() {
        if (!grid) return;
        grid.innerHTML = '';
        champs.forEach(name => {
            const b = document.createElement('div');
            b.className = 'champ-badge' + (pool.includes(name) ? ' pool-selected' : '');
            b.textContent = name;
            b.draggable = true;

            b.ondragstart = (e) => { dragCard = null; e.dataTransfer.setData('text/plain', name); };
            b.onclick = () => assign(activeSlot || Array.from(cards).find(c => !c.classList.contains('filled')), name);
            b.oncontextmenu = (e) => {
                e.preventDefault();
                pool = pool.includes(name) ? pool.filter(p => p !== name) : [...pool, name];
                b.classList.toggle('pool-selected', pool.includes(name));
                savePool();
                predict();
            };

            grid.appendChild(b);
        });
        filter();
    }

    function filter() {
        if (!grid) return;
        const q = search ? search.value.trim().toLowerCase() : '';
        Array.from(grid.children).forEach(b => {
            const isMatch = b.textContent.toLowerCase().includes(q);
            b.style.display = (isMatch && !b.classList.contains('picked')) ? 'block' : 'none';
        });
    }

    if (search) search.oninput = filter;

    cards.forEach(c => {
        c.draggable = true;
        c.onclick = (e) => {
            e.stopPropagation();
            if (e.target.closest('.target-btn')) {
                const isTarget = c.classList.contains('user-target');
                cards.forEach(x => x.classList.remove('user-target'));
                targetCard = isTarget ? null : c;
                if (!isTarget) c.classList.add('user-target');
                predict();
                return;
            }
            if (e.target.closest('.clear-btn')) return assign(c, 'Empty', true);

            cards.forEach(x => x.classList.remove('active-selection'));
            activeSlot = (activeSlot === c) ? null : c;
            if (activeSlot) c.classList.add('active-selection');
        };

        c.ondragstart = (e) => {
            if (!c.classList.contains('filled')) return e.preventDefault();
            dragCard = c;
            e.dataTransfer.setData('text/plain', c.querySelector('.champ-name').textContent);
        };

        c.ondragover = e => e.preventDefault();

        c.ondrop = (e) => {
            e.preventDefault();
            const val = e.dataTransfer.getData('text/plain');
            if (!val) return;

            const isTargetFilled = c.classList.contains('filled');
            const targetName = isTargetFilled ? c.querySelector('.champ-name').textContent : null;

            assign(c, val);
            if (dragCard && dragCard !== c) {
                assign(dragCard, isTargetFilled ? targetName : 'Empty', false);
            }
            dragCard = null;
        };
    });

    function assign(c, name, trigger = true) {
        if (!c) return;
        const oldName = c.querySelector('.champ-name').textContent;
        if (oldName && oldName !== 'Empty') togglePicked(oldName, false);

        const isFilled = name && name !== 'Empty';
        c.querySelector('.champ-name').textContent = isFilled ? name : 'Empty';
        c.classList.toggle('filled', isFilled);
        c.classList.remove('active-selection');

        if (isFilled) togglePicked(name, true);
        if (activeSlot === c) activeSlot = null;
        if (trigger) predict();
    }

    function togglePicked(name, isPicked) {
        if (!name || name === 'Empty' || !grid) return;
        const badge = Array.from(grid.children).find(b => b.textContent === name);
        if (badge) badge.classList.toggle('picked', isPicked);
        filter();
    }

    if (clearBtn) {
        clearBtn.onclick = () => {
            cards.forEach(c => {
                assign(c, 'Empty', false);
                c.classList.remove('user-target');
            });
            activeSlot = targetCard = null;
            bluePct.textContent = redPct.textContent = '50.0%';
            if (recBox) recBox.classList.add('hidden');
            filter();
        };
    }

    if (swapBtn) {
        swapBtn.onclick = () => {
            for (let i = 0; i < 5; i++) {
                const bName = cards[i].querySelector('.champ-name').textContent;
                const rName = cards[i + 5].querySelector('.champ-name').textContent;
                assign(cards[i], rName, false);
                assign(cards[i + 5], bName, false);
            }
            predict();
        };
    }

    function predict() {
        const payload = { blue_team: {}, red_team: {} };
        let bCount = 0, rCount = 0;

        cards.forEach(c => {
            const txt = c.querySelector('.champ-name').textContent;
            const val = txt === 'Empty' ? '' : txt;
            payload[c.dataset.team + '_team'][c.dataset.role] = val;
            if (val) (c.dataset.team === 'blue' ? bCount++ : rCount++);
        });

        if (!bCount || !rCount) {
            bluePct.textContent = redPct.textContent = '50.0%';
            if (recBox) recBox.classList.add('hidden');
            return;
        }

        post('/api/predict', payload).then(data => {
            if (data.error) return;
            const p = data.probability;
            bluePct.textContent = (p * 100).toFixed(1) + '%';
            redPct.textContent = ((1 - p) * 100).toFixed(1) + '%';

            if (targetCard && pool.length) {
                fetchRecommendations(payload, p);
            } else if (recBox) {
                recBox.classList.add('hidden');
            }
        });
    }

    function fetchRecommendations(payload, baseline) {
        const userBaseline = targetCard.dataset.team === 'blue' ? baseline : 1 - baseline;
        post('/api/recommend', {
            ...payload,
            user_side: targetCard.dataset.team,
            user_role: targetCard.dataset.role,
            candidates: pool
        }).then(data => {
            if (data.error || !data.recommendations || !recList || !recBox) return;
            recList.innerHTML = '';
            recBox.classList.remove('hidden');
            data.recommendations.slice(0, 5).forEach(rec => {
                const delta = (rec.win_rate - userBaseline) * 100;
                const sign = delta >= 0 ? '+' : '';
                const item = document.createElement('div');
                item.className = 'rec-item';
                item.innerHTML = `<span class="rec-name">${rec.name}</span> <span class="rec-diff">${(rec.win_rate * 100).toFixed(1)}% (${sign}${delta.toFixed(1)})</span>`;
                item.onclick = () => assign(targetCard, rec.name);
                recList.appendChild(item);
            });
        });
    }
});
