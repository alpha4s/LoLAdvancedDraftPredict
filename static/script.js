const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const BACKEND_URL = isLocal ? '' : 'https://loladvanceddraftpredict.onrender.com';

fetch(`${BACKEND_URL}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});

// Heartbeat ping every 10 minutes to keep Render awake while tab is open
setInterval(() => {
    fetch(`${BACKEND_URL}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
}, 10 * 60 * 1000);

document.addEventListener('DOMContentLoaded', () => {
    let championNames = [];
    let activeSlot = null;
    let userTargetCard = null;
    let currentTab = 'draft';
    let personalPool = [];

    try { personalPool = JSON.parse(localStorage.getItem('my_personal_champion_pool')) || []; } catch (e) {}

    const championGrid = document.getElementById('champion-grid');
    const poolSearch = document.getElementById('pool-search');
    const clearBtn = document.getElementById('clear-btn');
    const roleCards = document.querySelectorAll('.role-card');
    const selectAllBtn = document.getElementById('select-all-btn');
    const tabDraft = document.getElementById('tab-draft');
    const tabPool = document.getElementById('tab-pool');
    const recommendationsBox = document.getElementById('recommendations-box');
    const recommendationsList = document.getElementById('recommendations-list');

    const onboardingOverlay = document.getElementById('onboarding-overlay');
    const onboardGrid = document.getElementById('onboard-grid');
    const onboardSearch = document.getElementById('onboard-search');
    const onboardSelectAll = document.getElementById('onboard-select-all');
    const onboardSkipBtn = document.getElementById('onboard-skip-btn');
    const onboardStartBtn = document.getElementById('onboard-start-btn');

    fetch('champions.json')
        .then(res => res.json())
        .then(data => {
            championNames = Object.values(data).sort();
            if (!localStorage.getItem('draft_predictor_onboarded')) {
                onboardingOverlay.classList.remove('hidden');
                renderGrid(onboardGrid, true);
                setupOnboardingListeners();
            }
            renderGrid(championGrid, false);
            triggerLivePrediction();
        })
        .catch(err => console.error(err));

    const savePool = () => {
        try { localStorage.setItem('my_personal_champion_pool', JSON.stringify(personalPool)); } catch (e) {}
    };

    function renderGrid(container, isPoolManager) {
        container.innerHTML = '';
        championNames.forEach(name => {
            const badge = document.createElement('div');
            badge.className = 'champ-badge';
            badge.textContent = name;
            badge.dataset.name = name;

            if (personalPool.includes(name)) badge.classList.add('pool-selected');

            badge.addEventListener('click', () => {
                if (isPoolManager || currentTab === 'pool') {
                    const idx = personalPool.indexOf(name);
                    if (idx > -1) {
                        personalPool.splice(idx, 1);
                        badge.classList.remove('pool-selected');
                    } else {
                        personalPool.push(name);
                        badge.classList.add('pool-selected');
                    }
                    savePool();
                    updateSelectAllButtons();
                    triggerLivePrediction();
                } else {
                    const targetCard = activeSlot || Array.from(roleCards).find(c => !c.classList.contains('filled'));
                    assignChampion(targetCard, name);
                }
            });

            if (!isPoolManager) {
                badge.setAttribute('draggable', 'true');
                badge.addEventListener('dragstart', (e) => {
                    if (currentTab === 'pool') return e.preventDefault();
                    e.dataTransfer.setData('text/plain', name);
                    roleCards.forEach(c => { if (!c.classList.contains('filled')) c.classList.add('drag-possible'); });
                });
                badge.addEventListener('dragend', () => roleCards.forEach(c => c.classList.remove('drag-possible')));
            }
            container.appendChild(badge);
        });
        applyFilters();
    }

    function filterGrid(input, grid, hidePicked) {
        const val = input.value.trim().toLowerCase();
        grid.querySelectorAll('.champ-badge').forEach(badge => {
            const match = badge.textContent.toLowerCase().includes(val);
            const picked = hidePicked && badge.classList.contains('picked');
            badge.style.display = (match && !picked) ? 'block' : 'none';
        });
    }

    const applyFilters = () => {
        filterGrid(poolSearch, championGrid, currentTab === 'draft');
        if (onboardGrid.children.length > 0) filterGrid(onboardSearch, onboardGrid, false);
    };

    poolSearch.addEventListener('input', applyFilters);

    function updateSelectAllButtons() {
        const isFull = personalPool.length >= championNames.length;
        selectAllBtn.textContent = onboardSelectAll.textContent = isFull ? 'DESELECT ALL' : 'SELECT ALL';
    }

    tabDraft.addEventListener('click', () => {
        currentTab = 'draft';
        tabDraft.classList.add('active');
        tabPool.classList.remove('active');
        poolTitle.textContent = 'CHAMPION LIST';
        selectAllBtn.classList.add('hidden');
        applyFilters();
    });

    tabPool.addEventListener('click', () => {
        currentTab = 'pool';
        tabPool.classList.add('active');
        tabDraft.classList.remove('active');
        poolTitle.textContent = 'MANAGE MY POOL';
        selectAllBtn.classList.remove('hidden');
        updateSelectAllButtons();
        applyFilters();
    });

    const toggleAllPool = (select) => {
        personalPool = select ? [...championNames] : [];
        const isOnboarding = !onboardingOverlay.classList.contains('hidden');
        const activeGrid = isOnboarding ? onboardGrid : championGrid;
        activeGrid.querySelectorAll('.champ-badge').forEach(b => b.classList.toggle('pool-selected', select));
        savePool();
        updateSelectAllButtons();
        triggerLivePrediction();
    };

    selectAllBtn.addEventListener('click', () => toggleAllPool(personalPool.length < championNames.length));
    onboardSelectAll.addEventListener('click', () => toggleAllPool(personalPool.length < championNames.length));

    roleCards.forEach(card => {
        card.addEventListener('click', (e) => {
            e.stopPropagation();
            if (e.target.classList.contains('target-btn')) {
                const alreadyTarget = card.classList.contains('user-target');
                roleCards.forEach(c => c.classList.remove('user-target'));
                userTargetCard = alreadyTarget ? null : card;
                if (!alreadyTarget) card.classList.add('user-target');
                triggerLivePrediction();
                return;
            }
            if (e.target.classList.contains('clear-btn')) return clearSlot(card);

            roleCards.forEach(c => c.classList.remove('active-selection'));
            if (activeSlot === card) {
                activeSlot = null;
            } else {
                activeSlot = card;
                card.classList.add('active-selection');
            }
        });

        card.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (currentTab === 'draft') card.classList.add('drag-over');
        });
        card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
        card.addEventListener('drop', (e) => {
            e.preventDefault();
            card.classList.remove('drag-over');
            const name = e.dataTransfer.getData('text/plain');
            if (name && currentTab === 'draft') assignChampion(card, name);
        });
    });

    function assignChampion(card, name) {
        if (!card) return;
        if (card.classList.contains('filled')) {
            const oldName = card.querySelector('.champ-name').textContent;
            togglePoolChampion(oldName, false);
        }
        roleCards.forEach(c => {
            if (c !== card && c.classList.contains('filled') && c.querySelector('.champ-name').textContent === name) clearSlot(c);
        });
        card.querySelector('.champ-name').textContent = name;
        card.classList.add('filled');
        card.classList.remove('active-selection');
        togglePoolChampion(name, true);
        if (activeSlot === card) activeSlot = null;
        triggerLivePrediction();
    }

    function clearSlot(card) {
        if (!card || !card.classList.contains('filled')) return;
        const name = card.querySelector('.champ-name').textContent;
        card.querySelector('.champ-name').textContent = 'Empty';
        card.classList.remove('filled', 'active-selection');
        togglePoolChampion(name, false);
        if (activeSlot === card) activeSlot = null;
        triggerLivePrediction();
    }

    function togglePoolChampion(name, isPicked) {
        const badge = championGrid.querySelector(`.champ-badge[data-name="${name}"]`);
        if (badge) {
            badge.classList.toggle('picked', isPicked);
            applyFilters();
        }
    }

    function triggerLivePrediction() {
        const getVal = (team, role) => {
            const card = document.querySelector(`.role-card[data-team="${team}"][data-role="${role}"]`);
            return card.classList.contains('filled') ? card.querySelector('.champ-name').textContent : '';
        };

        const roles = ['top', 'jungle', 'mid', 'bot', 'support'];
        const payload = {
            blue_team: roles.reduce((acc, r) => ({ ...acc, [r]: getVal('blue', r) }), {}),
            red_team: roles.reduce((acc, r) => ({ ...acc, [r]: getVal('red', r) }), {})
        };

        const blueCount = document.querySelectorAll('.role-card[data-team="blue"].filled').length;
        const redCount = document.querySelectorAll('.role-card[data-team="red"].filled').length;

        if (blueCount < 2 || redCount < 2) {
            document.getElementById('blue-percent').textContent = '50.0%';
            document.getElementById('red-percent').textContent = '50.0%';
            document.getElementById('blue-bar').style.width = '50%';
            recommendationsBox.classList.add('hidden');
            return;
        }

        fetch(`${BACKEND_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(res => res.json())
        .then(data => {
            if (data.error) return;
            const prob = data.probability;
            document.getElementById('blue-percent').textContent = (prob * 100).toFixed(1) + '%';
            document.getElementById('red-percent').textContent = ((1 - prob) * 100).toFixed(1) + '%';
            document.getElementById('blue-bar').style.width = (prob * 100) + '%';

            if (userTargetCard && personalPool.length > 0) {
                const userSide = userTargetCard.getAttribute('data-team');
                const userRole = userTargetCard.getAttribute('data-role');

                fetch(`${BACKEND_URL}/api/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ...payload,
                        user_side: userSide,
                        user_role: userRole,
                        candidates: personalPool
                    })
                })
                .then(res => res.json())
                .then(recData => {
                    if (recData.error) return;
                    recommendationsList.innerHTML = '';
                    recommendationsBox.classList.remove('hidden');

                    const currentWinRate = userSide === 'blue' ? prob : (1.0 - prob);
                    recData.recommendations.forEach((item, index) => {
                        const delta = item.win_rate - currentWinRate;
                        const isPos = delta >= 0;
                        const row = document.createElement('div');
                        row.className = `recommendation-row ${isPos ? 'positive' : 'negative'}`;
                        row.innerHTML = `
                            <span>${index + 1}. ${item.name}</span>
                            <span>
                                <span class="rec-winrate">${(item.win_rate * 100).toFixed(1)}%</span>
                                <span class="rec-delta ${isPos ? 'plus' : 'minus'}">${isPos ? '+' : ''}${(delta * 100).toFixed(1)}%</span>
                            </span>
                        `;
                        recommendationsList.appendChild(row);
                    });
                })
                .catch(err => console.error(err));
            } else {
                recommendationsBox.classList.add('hidden');
            }
        })
        .catch(err => console.error(err));
    }

    clearBtn.addEventListener('click', () => {
        roleCards.forEach(clearSlot);
        roleCards.forEach(c => c.classList.remove('user-target'));
        userTargetCard = null;
        poolSearch.value = '';
        applyFilters();
        triggerLivePrediction();
    });

    function setupOnboardingListeners() {
        onboardSearch.addEventListener('input', applyFilters);

        const closeOnboarding = () => {
            localStorage.setItem('draft_predictor_onboarded', 'true');
            onboardingOverlay.classList.add('hidden');
            renderGrid(championGrid, false);
            triggerLivePrediction();
        };

        onboardSkipBtn.addEventListener('click', () => {
            personalPool = [];
            savePool();
            closeOnboarding();
        });
        onboardStartBtn.addEventListener('click', closeOnboarding);
    }
});
