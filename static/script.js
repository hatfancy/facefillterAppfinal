document.addEventListener('DOMContentLoaded', () => {
    const filterBtns = document.querySelectorAll('.filter-btn');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button styling
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Tell the server which filter to use
            const filter = btn.getAttribute('data-filter');
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
});
