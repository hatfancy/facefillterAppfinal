document.addEventListener('DOMContentLoaded', () => {
    // ── Elements ──
    const filterBtns = document.querySelectorAll('.filter-btn');
    const captureBtn = document.getElementById('btn-capture');
    const videoContainer = document.getElementById('video-container');
    const filterLabel = document.getElementById('filter-label');
    const captureModal = document.getElementById('capture-modal');
    const modalBackdrop = document.getElementById('modal-backdrop');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const capturedImage = document.getElementById('captured-image');
    const downloadBtn = document.getElementById('download-btn');
    const postBtn = document.getElementById('post-btn');
    const captionInput = document.getElementById('caption-input');
    const galleryGrid = document.getElementById('gallery-grid');
    const galleryEmpty = document.getElementById('gallery-empty');
    const galleryCount = document.getElementById('gallery-count');
    const toast = document.getElementById('toast');

    // Tab elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const panelCamera = document.getElementById('panel-camera');
    const panelGallery = document.getElementById('panel-gallery');

    let currentFilter = 'none';
    let lastCapturedImage = null;

    // ── Filter names for the overlay label ──
    const filterNames = {
        'none':       'No Filter',
        'smiley':     '😊 Smiley',
        'wink':       '😉 Wink',
        'cool':       '😎 Cool Guy',
        'heart_eyes': '😍 Heart Eyes',
        'clown':      '🤡 Clown',
        'ghost':      '👻 Ghost',
        'pumpkin':    '🎃 Pumpkin',
        'devil':      '😈 Devil',
        'pixel_art':  '👾 Pixel Art',
        'neon_glow':  '💜 Neon Glow',
        'sketch':     '✏️ Sketch',
        'swirl':      '🌀 Swirl',
        'thermal':    '🔥 Thermal',
        'glitch':     '📺 Glitch',
        'pop_art':    '🎨 Pop Art',
        'mirror':     '🪞 Mirror',
        'emboss':     '🗿 Emboss',
        'watercolor': '🖌️ Watercolor',
    };

    // ── Tab switching ──
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const tab = btn.dataset.tab;
            panelCamera.classList.toggle('active', tab === 'camera');
            panelGallery.classList.toggle('active', tab === 'gallery');

            if (tab === 'gallery') {
                loadGallery();
            }
        });
    });

    // ── Filter selection ──
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const filter = btn.getAttribute('data-filter');
            currentFilter = filter;
            filterLabel.textContent = filterNames[filter] || filter;

            fetch('/set_filter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filter: filter })
            })
            .then(res => res.json())
            .then(data => console.log('Filter set to:', data.filter))
            .catch(err => console.error('Error setting filter:', err));
        });
    });

    // ── Capture / Screenshot ──
    captureBtn.addEventListener('click', () => {
        captureBtn.classList.add('capturing');
        videoContainer.classList.add('flash');
        setTimeout(() => {
            captureBtn.classList.remove('capturing');
            videoContainer.classList.remove('flash');
        }, 500);

        fetch('/capture', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                lastCapturedImage = data.image;
                capturedImage.src = data.image;
                downloadBtn.href = data.image;

                const now = new Date();
                const ts = now.toISOString().replace(/[:.]/g, '-').slice(0, 19);
                downloadBtn.download = `face_filter_${ts}.jpg`;

                captionInput.value = '';
                captureModal.classList.add('visible');
            } else {
                console.error('Capture failed:', data.message);
            }
        })
        .catch(err => console.error('Capture error:', err));
    });

    // ── Post to Gallery ──
    postBtn.addEventListener('click', () => {
        if (!lastCapturedImage) return;

        const caption = captionInput.value.trim();

        postBtn.disabled = true;
        postBtn.textContent = 'Posting…';

        fetch('/post_capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: lastCapturedImage,
                filter: currentFilter,
                caption: caption
            })
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'ok') {
                closeModal();
                showToast('🎉 Posted to Gallery!');
                updateGalleryCount();
            }
        })
        .catch(err => console.error('Post error:', err))
        .finally(() => {
            postBtn.disabled = false;
            postBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                Post to Gallery
            `;
        });
    });

    // ── Modal close ──
    function closeModal() {
        captureModal.classList.remove('visible');
    }

    modalCloseBtn.addEventListener('click', closeModal);
    modalBackdrop.addEventListener('click', closeModal);

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && captureModal.classList.contains('visible')) {
            closeModal();
        }
    });

    // ── Toast ──
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('visible');
        setTimeout(() => toast.classList.remove('visible'), 2500);
    }

    // ── Gallery ──
    function loadGallery() {
        fetch('/gallery_feed')
        .then(res => res.json())
        .then(data => {
            renderGallery(data.posts);
        })
        .catch(err => console.error('Gallery load error:', err));
    }

    function renderGallery(posts) {
        galleryGrid.innerHTML = '';

        if (posts.length === 0) {
            galleryEmpty.classList.remove('hidden');
            return;
        }

        galleryEmpty.classList.add('hidden');

        posts.forEach(post => {
            const card = document.createElement('div');
            card.className = 'gallery-card';
            card.innerHTML = `
                <img class="gallery-card-image" src="${post.image}" alt="Gallery photo" loading="lazy" />
                <div class="gallery-card-body">
                    ${post.caption ? `<div class="gallery-card-caption">${escapeHtml(post.caption)}</div>` : ''}
                    <div class="gallery-card-meta">
                        <span class="gallery-card-filter">${filterNames[post.filter] || post.filter}</span>
                        <span class="gallery-card-time">${post.timestamp}</span>
                    </div>
                </div>
                <div class="gallery-card-actions">
                    <button class="like-btn" data-id="${post.id}">
                        <span class="like-icon">♡</span>
                        <span class="like-count">${post.likes}</span>
                    </button>
                    <a class="gallery-download-btn" href="${post.image}" download="gallery_${post.id}.jpg">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                        Save
                    </a>
                </div>
            `;

            // Like button handler
            const likeBtn = card.querySelector('.like-btn');
            likeBtn.addEventListener('click', () => {
                fetch('/like_post', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id: post.id })
                })
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'ok') {
                        likeBtn.classList.add('liked');
                        likeBtn.querySelector('.like-icon').textContent = '♥';
                        likeBtn.querySelector('.like-count').textContent = data.likes;
                    }
                });
            });

            galleryGrid.appendChild(card);
        });
    }

    function updateGalleryCount() {
        fetch('/gallery_feed')
        .then(res => res.json())
        .then(data => {
            galleryCount.textContent = data.posts.length;
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Initial gallery count
    updateGalleryCount();
});
