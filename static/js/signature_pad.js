function initSignaturePad(canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');

    // Enhanced drawing settings
    ctx.strokeStyle = "#2d3436";
    ctx.lineWidth = 2.5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.shadowBlur = 1;
    ctx.shadowColor = "rgba(0,0,0,0.1)";

    let drawing = false;
    let lastX = 0;
    let lastY = 0;
    let points = [];

    // Initialize with white background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        if (e.touches && e.touches.length > 0) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY
            };
        } else {
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }
    }

    function startDraw(e) {
        drawing = true;
        const pos = getPos(e);
        lastX = pos.x;
        lastY = pos.y;
        points = [{x: pos.x, y: pos.y}];
        
        // Add visual feedback
        canvas.style.borderColor = "#667eea";
        e.preventDefault();
    }

    function draw(e) {
        if (!drawing) return;
        
        const pos = getPos(e);
        points.push({x: pos.x, y: pos.y});
        
        // Smooth line drawing
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        
        // Use quadratic curve for smoother lines
        const midX = (lastX + pos.x) / 2;
        const midY = (lastY + pos.y) / 2;
        ctx.quadraticCurveTo(lastX, lastY, midX, midY);
        
        ctx.stroke();
        
        lastX = pos.x;
        lastY = pos.y;
        e.preventDefault();
    }

    function endDraw(e) {
        if (drawing) {
            drawing = false;
            // Reset border color
            canvas.style.borderColor = "#e1e5e9";
        }
        e.preventDefault();
    }

    // Mouse events
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);

    // Touch events for mobile
    canvas.addEventListener('touchstart', startDraw, {passive: false});
    canvas.addEventListener('touchmove', draw, {passive: false});
    canvas.addEventListener('touchend', endDraw, {passive: false});
    canvas.addEventListener('touchcancel', endDraw, {passive: false});
    
    // Prevent scrolling when drawing on mobile
    canvas.addEventListener('touchmove', (e) => {
        if (drawing) {
            e.preventDefault();
        }
    }, {passive: false});
}

function clearPad(canvasId) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    
    // Clear with animation effect
    canvas.style.opacity = '0.5';
    setTimeout(() => {
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        canvas.style.opacity = '1';
    }, 100);
}

// Add smooth transition when canvas loads
document.addEventListener('DOMContentLoaded', () => {
    const canvases = document.querySelectorAll('.signature-pad');
    canvases.forEach(canvas => {
        canvas.style.transition = 'all 0.3s ease';
    });
});