// Completed Booklets Viewer JavaScript

const ARC_COLORS = {
    0: '#000000', 1: '#0074d9', 2: '#ff4136', 3: '#2ecc40', 4: '#ffdc00',
    5: '#ff851b', 6: '#f012be', 7: '#7fdbff', 8: '#870c25', 9: '#9577cd'
};

let viewerState = {
    puzzles: [],
    currentPuzzle: null
};

document.addEventListener('DOMContentLoaded', () => {
    loadCompletedPuzzles();
    initializeSearch();
});

async function loadCompletedPuzzles() {
    try {
        const response = await fetch('/api/completed_puzzles');
        const data = await response.json();
        viewerState.puzzles = data.puzzles;
        displayPuzzleList();
    } catch (error) {
        console.error('Error loading puzzles:', error);
    }
}

function displayPuzzleList() {
    const list = document.getElementById('puzzle-list');
    
    if (viewerState.puzzles.length === 0) {
        list.innerHTML = '<p style="color: #bdc3c7;">No completed puzzles yet. Create some steps first!</p>';
        return;
    }
    
    let html = '';
    viewerState.puzzles.forEach(puzzle => {
        const totalSteps = puzzle.training_steps + puzzle.testing_steps;
        html += `
            <div class="puzzle-item" data-puzzle-id="${puzzle.id}">
                <strong>${puzzle.id}</strong>
                <div style="font-size: 0.85rem; color: #ecf0f1; margin-top: 0.3rem;">
                    📚 ${puzzle.training_examples} training, ${puzzle.testing_examples} testing
                    <br>📄 ${totalSteps} total steps
                </div>
            </div>
        `;
    });
    
    list.innerHTML = html;
    
    // Add click handlers
    document.querySelectorAll('.puzzle-item').forEach(item => {
        item.addEventListener('click', () => {
            const puzzleId = item.dataset.puzzleId;
            selectPuzzle(puzzleId);
        });
    });
}

function initializeSearch() {
    document.getElementById('search-puzzle').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        document.querySelectorAll('.puzzle-item').forEach(item => {
            const puzzleId = item.dataset.puzzleId.toLowerCase();
            if (puzzleId.includes(query)) {
                item.style.display = 'block';
            } else {
                item.style.display = 'none';
            }
        });
    });
}

async function selectPuzzle(puzzleId) {
    viewerState.currentPuzzle = puzzleId;
    
    // Update active state
    document.querySelectorAll('.puzzle-item').forEach(item => {
        if (item.dataset.puzzleId === puzzleId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    // Load puzzle details
    await loadPuzzleDetails(puzzleId);
}

async function loadPuzzleDetails(puzzleId) {
    try {
        const response = await fetch(`/api/completed_puzzle/${puzzleId}`);
        const data = await response.json();
        
        displayPuzzleDetails(data);
        displayBooklet(data);
    } catch (error) {
        console.error('Error loading puzzle details:', error);
    }
}

function displayPuzzleDetails(data) {
    const details = document.getElementById('current-puzzle-details');
    
    const totalSteps = data.training.reduce((sum, ex) => sum + ex.steps.length, 0) +
                      data.testing.reduce((sum, ex) => sum + ex.steps.length, 0);
    
    details.innerHTML = `
        <strong>Puzzle:</strong> <code>${data.puzzle_id}</code><br>
        <strong>Training Examples:</strong> ${data.training.length}<br>
        <strong>Testing Examples:</strong> ${data.testing.length}<br>
        <strong>Total Steps:</strong> ${totalSteps}<br>
        <strong>Universal Steps:</strong> ${data.universal_steps.length}
    `;
}

function displayBooklet(data) {
    const display = document.getElementById('booklet-display');
    
    let html = `
        <!-- Summary Section -->
        <div class="summary-section">
            <h2>📋 Puzzle ${data.puzzle_id}</h2>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">${data.training.length}</span>
                    <span class="stat-label">Training Examples</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${data.testing.length}</span>
                    <span class="stat-label">Testing Examples</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${data.training.reduce((s, e) => s + e.steps.length, 0) + data.testing.reduce((s, e) => s + e.steps.length, 0)}</span>
                    <span class="stat-label">Total Steps</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">${data.universal_steps.length}</span>
                    <span class="stat-label">Universal Templates</span>
                </div>
            </div>
        </div>
    `;
    
    // Display training examples
    if (data.training.length > 0) {
        data.training.forEach(example => {
            html += renderExample(example, 'training', data.puzzle_id);
        });
    }
    
    // Display testing examples
    if (data.testing.length > 0) {
        data.testing.forEach(example => {
            html += renderExample(example, 'testing', data.puzzle_id);
        });
    }
    
    if (data.training.length === 0 && data.testing.length === 0) {
        html += '<div class="no-data-message"><h3>No steps found for this puzzle</h3></div>';
    }
    
    display.innerHTML = html;
    
    // Add edit button handlers
    document.querySelectorAll('.btn-edit-example').forEach(btn => {
        btn.addEventListener('click', () => {
            const exType = btn.dataset.exampleType;
            const exNum = btn.dataset.exampleNum;
            const puzzleId = btn.dataset.puzzleId;
            editExample(puzzleId, exType, exNum);
        });
    });
    
    document.querySelectorAll('.btn-edit-step-viewer').forEach(btn => {
        btn.addEventListener('click', () => {
            const exType = btn.dataset.exampleType;
            const exNum = btn.dataset.exampleNum;
            const stepNum = btn.dataset.stepNum;
            const puzzleId = btn.dataset.puzzleId;
            editStep(puzzleId, exType, exNum, stepNum);
        });
    });
}

function renderExample(example, type, puzzleId) {
    const badgeClass = type === 'training' ? 'example-badge-viewer' : 'example-badge-viewer testing';
    
    let html = `
        <div class="example-section">
            <div class="example-header">
                <div>
                    <h3 class="example-title">${type.toUpperCase()} Example ${String(example.number).padStart(2, '0')}</h3>
                    <span class="${badgeClass}">${example.steps.length} steps</span>
                </div>
                <button class="btn-edit-example" 
                        data-puzzle-id="${puzzleId}"
                        data-example-type="${type}" 
                        data-example-num="${example.number}">
                    ✏️ Edit This Example
                </button>
            </div>
            
            <div class="steps-container">
    `;
    
    example.steps.forEach(step => {
        const timestamp = step.timestamp ? new Date(step.timestamp).toLocaleString() : '';
        
        html += `
            <div class="step-card">
                <div class="step-card-header">
                    <span class="step-number">Step ${String(step.step_num).padStart(2, '0')}</span>
                    <span class="step-name">${step.step_name || 'Unnamed'}</span>
                    <span class="step-timestamp">${timestamp}</span>
                </div>
                
                <div class="step-grid-display">
                    <div class="grid-display" id="grid-${type}-${example.number}-${step.step_num}"></div>
                    <p class="grid-size">Size: ${step.grid.length}×${step.grid[0].length}</p>
                </div>
                
                <div class="step-description">
                    ${step.description || 'No description'}
                </div>
                
                <div class="step-actions">
                    <button class="btn-edit-step-viewer" 
                            data-puzzle-id="${puzzleId}"
                            data-example-type="${type}" 
                            data-example-num="${example.number}"
                            data-step-num="${step.step_num}">
                        ✏️ Edit Step
                    </button>
                </div>
            </div>
        `;
    });
    
    html += `
            </div>
        </div>
    `;
    
    // Render grids after HTML is added
    setTimeout(() => {
        example.steps.forEach(step => {
            renderGrid(step.grid, `grid-${type}-${example.number}-${step.step_num}`);
        });
    }, 0);
    
    return html;
}

function renderGrid(grid, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const gridDiv = document.createElement('div');
    gridDiv.style.display = 'inline-block';
    
    grid.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.style.display = 'flex';
        
        row.forEach(value => {
            const cell = document.createElement('div');
            cell.style.width = '25px';
            cell.style.height = '25px';
            cell.style.backgroundColor = ARC_COLORS[value];
            cell.style.border = '1px solid #666';
            rowDiv.appendChild(cell);
        });
        
        gridDiv.appendChild(rowDiv);
    });
    
    container.innerHTML = '';
    container.appendChild(gridDiv);
}

function editExample(puzzleId, exampleType, exampleNum) {
    // Redirect to main app with parameters
    window.location.href = `/?puzzle=${puzzleId}&type=${exampleType}&num=${exampleNum}`;
}

function editStep(puzzleId, exampleType, exampleNum, stepNum) {
    // Redirect to main app with parameters to edit specific step
    window.location.href = `/?puzzle=${puzzleId}&type=${exampleType}&num=${exampleNum}&edit_step=${stepNum}`;
}

