export class UIController {
    constructor(onDraftChange) {
        this.onDraftChange = onDraftChange;
        this.activeSlot = null;
        this.targetCard = null;
        this.dragCard = null;
        this.pool = [];
        this.champs = [];

        try {
            const savedPool = JSON.parse(localStorage.getItem('my_personal_champion_pool'));
            this.pool = Array.isArray(savedPool) ? savedPool : [];
        } catch (e) {}

        this.$ = id => document.getElementById(id);
        this.cards = document.querySelectorAll('.role-card');
        this.grid = this.$('champion-grid');
        this.search = this.$('pool-search');
        this.clearBtn = this.$('clear-btn');
        this.swapBtn = this.$('swap-teams-btn');
        this.recBox = this.$('recommendations-box');
        this.recList = this.$('recommendations-list');
        this.bluePct = this.$('blue-percent');
        this.redPct = this.$('red-percent');
        this.status = this.$('model-status');

        this.bindEvents();
    }

    savePool() {
        try {
            localStorage.setItem('my_personal_champion_pool', JSON.stringify(this.pool));
        } catch (e) {}
    }

    setStatus(message, state = 'ready') {
        if (!this.status) return;
        this.status.textContent = message;
        this.status.className = `model-status ${state}`;
    }

    bindEvents() {
        if (this.search) {
            this.search.oninput = () => this.filter();
        }

        this.cards.forEach(c => {
            c.draggable = true;
            c.onclick = (e) => {
                e.stopPropagation();
                if (e.target.closest('.target-btn')) {
                    const isTarget = c.classList.contains('user-target');
                    this.cards.forEach(x => x.classList.remove('user-target'));
                    this.targetCard = isTarget ? null : c;
                    if (!isTarget) c.classList.add('user-target');
                    this.onDraftChange();
                    return;
                }
                if (e.target.closest('.clear-btn')) return this.assign(c, 'Empty', true);

                this.cards.forEach(x => x.classList.remove('active-selection'));
                this.activeSlot = (this.activeSlot === c) ? null : c;
                if (this.activeSlot) c.classList.add('active-selection');
                this.onDraftChange();
            };

            c.ondragstart = (e) => {
                if (!c.classList.contains('filled')) return e.preventDefault();
                this.dragCard = c;
                e.dataTransfer.setData('text/plain', c.querySelector('.champ-name').textContent);
            };

            c.ondragover = e => e.preventDefault();

            c.ondrop = (e) => {
                e.preventDefault();
                const val = e.dataTransfer.getData('text/plain');
                if (!val) return;

                const isTargetFilled = c.classList.contains('filled');
                const targetName = isTargetFilled ? c.querySelector('.champ-name').textContent : null;

                this.assign(c, val, false);
                if (this.dragCard && this.dragCard !== c) {
                    this.assign(this.dragCard, isTargetFilled ? targetName : 'Empty', false);
                }
                this.dragCard = null;
                this.onDraftChange();
            };
        });

        if (this.clearBtn) {
            this.clearBtn.onclick = () => {
                this.cards.forEach(c => {
                    this.assign(c, 'Empty', false);
                    c.classList.remove('user-target');
                });
                this.activeSlot = this.targetCard = null;
                this.bluePct.textContent = this.redPct.textContent = '50.0%';
                if (this.recBox) this.recBox.classList.add('hidden');
                this.filter();
                this.onDraftChange();
            };
        }

        if (this.swapBtn) {
            this.swapBtn.onclick = () => {
                for (let i = 0; i < 5; i++) {
                    const bName = this.cards[i].querySelector('.champ-name').textContent;
                    const rName = this.cards[i + 5].querySelector('.champ-name').textContent;
                    this.assign(this.cards[i], rName, false);
                    this.assign(this.cards[i + 5], bName, false);
                }
                if (this.targetCard) {
                    const idx = Array.from(this.cards).indexOf(this.targetCard);
                    this.cards.forEach(x => x.classList.remove('user-target'));
                    this.targetCard = this.cards[(idx + 5) % 10];
                    if (this.targetCard) this.targetCard.classList.add('user-target');
                }
                this.onDraftChange();
            };
        }
    }

    renderGrid(champs) {
        this.champs = champs;
        if (!this.grid) return;
        this.grid.innerHTML = '';
        champs.forEach(name => {
            const b = document.createElement('div');
            b.className = 'champ-badge' + (this.pool.includes(name) ? ' pool-selected' : '');
            b.textContent = name;
            b.draggable = true;

            b.ondragstart = (e) => { this.dragCard = null; e.dataTransfer.setData('text/plain', name); };
            b.onclick = () => this.assign(this.activeSlot || Array.from(this.cards).find(c => !c.classList.contains('filled')), name);
            b.oncontextmenu = (e) => {
                e.preventDefault();
                this.pool = this.pool.includes(name) ? this.pool.filter(p => p !== name) : [...this.pool, name];
                b.classList.toggle('pool-selected', this.pool.includes(name));
                this.savePool();
                this.onDraftChange();
            };

            this.grid.appendChild(b);
        });
        this.filter();
    }

    filter() {
        if (!this.grid) return;
        const q = this.search ? this.search.value.trim().toLowerCase() : '';
        Array.from(this.grid.children).forEach(b => {
            const isMatch = b.textContent.toLowerCase().includes(q);
            b.style.display = (isMatch && !b.classList.contains('picked')) ? 'block' : 'none';
        });
    }

    assign(c, name, trigger = true) {
        if (!c) return;
        const oldName = c.querySelector('.champ-name').textContent;
        if (oldName && oldName !== 'Empty') this.togglePicked(oldName, false);

        const isFilled = name && name !== 'Empty';
        c.querySelector('.champ-name').textContent = isFilled ? name : 'Empty';
        c.classList.toggle('filled', isFilled);
        c.classList.remove('active-selection');

        if (isFilled) this.togglePicked(name, true);
        if (this.activeSlot === c) this.activeSlot = null;
        if (trigger) this.onDraftChange();
    }

    togglePicked(name, isPicked) {
        if (!name || name === 'Empty' || !this.grid) return;
        const badge = Array.from(this.grid.children).find(b => b.textContent === name);
        if (badge) badge.classList.toggle('picked', isPicked);
        this.filter();
    }

    updateWinRates(blueProb) {
        const bPct = (blueProb * 100).toFixed(1);
        const rPct = ((1 - blueProb) * 100).toFixed(1);
        this.bluePct.textContent = `${bPct}%`;
        this.redPct.textContent = `${rPct}%`;
    }

    renderRecommendations(recs) {
        if (!this.recBox) return;
        if (!recs || recs.length === 0) {
            this.recBox.classList.add('hidden');
            return;
        }

        this.recBox.classList.remove('hidden');
        this.recList.innerHTML = '';
        recs.forEach(r => {
            const div = document.createElement('div');
            div.className = 'rec-item';
            const deltaPct = Number(r.delta) * 100;
            const safeDelta = Number.isFinite(deltaPct) ? deltaPct : 0;
            div.innerHTML = `
                <span class="rec-name">${r.name}</span>
                <span class="rec-diff ${safeDelta >= 0 ? 'positive' : 'negative'}">${safeDelta >= 0 ? '+' : ''}${safeDelta.toFixed(2)} pp</span>
            `;
            div.onclick = () => {
                if (this.targetCard) {
                    this.assign(this.targetCard, r.name);
                }
            };
            this.recList.appendChild(div);
        });
    }

    getSlotName(team, role) {
        const c = document.querySelector(`.role-card[data-team="${team}"][data-role="${role}"]`);
        const txt = c ? c.querySelector('.champ-name').textContent : 'Empty';
        return txt === 'Empty' ? '' : txt;
    }
}
