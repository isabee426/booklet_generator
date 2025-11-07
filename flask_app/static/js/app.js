// ARC Puzzle Creator - Main JavaScript

// ARC Colors
const ARC_COLORS = {
    0: '#000000', 1: '#0074d9', 2: '#ff4136', 3: '#2ecc40', 4: '#ffdc00',
    5: '#ff851b', 6: '#f012be', 7: '#7fdbff', 8: '#870c25', 9: '#9577cd'
};

const COLOR_NAMES = {
    0: 'Black', 1: 'Blue', 2: 'Red', 3: 'Green', 4: 'Yellow',
    5: 'Orange', 6: 'Magenta', 7: 'Cyan', 8: 'Maroon', 9: 'Purple'
};

// State
let state = {
    tasks: { training: [], evaluation: [] },
    currentDataset: 'training',
    currentTaskId: null,
    currentTaskData: null,
    exampleType: 'training',
    exampleNum: 1,
    selectedColor: 1,
    grid: [],
    originalInput: [],
    originalOutput: null,
    universalSteps: [],
    currentStepIndex: 0,
    lastStepGrid: null,
    // Drawing tools
    drawingTool: 'paint',  // 'paint', 'fill', 'select'
    isSelecting: false,
    selectionStart: null,
    selectionEnd: null,
    selectedCells: new Set(),
    // Edit mode
    editMode: false,
    editingStep: null
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadTasks();
    initializeColorPalette();
    
    // Check for URL parameters (from viewer)
    const params = new URLSearchParams(window.location.search);
    if (params.has('puzzle')) {
        handleURLParameters(params);
    }
});

function handleURLParameters(params) {
    const puzzleId = params.get('puzzle');
    const exampleType = params.get('type') || 'training';
    const exampleNum = parseInt(params.get('num')) || 1;
    const editStep = params.get('edit_step');
    
    // Wait for tasks to load, then navigate
    setTimeout(() => {
        // Set state
        state.currentTaskId = puzzleId;
        state.exampleType = exampleType;
        state.exampleNum = exampleNum;
        
        // Update UI
        document.querySelector(`input[name="example-type"][value="${exampleType}"]`).checked = true;
        
        // Load task
        loadTask().then(() => {
            if (editStep) {
                // Find and edit specific step
                loadSteps().then(() => {
                    // Will need to implement step editing from URL
                    console.log(`Navigate to edit step ${editStep}`);
                });
            }
        });
    }, 500);
}

function initializeEventListeners() {
    // Dataset selection
    document.querySelectorAll('input[name="dataset"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.currentDataset = e.target.value;
            populatePuzzleSelect();
        });
    });
    
    // Example type selection
    document.querySelectorAll('input[name="example-type"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.exampleType = e.target.value;
            populateExampleSelect();
            loadExample();
        });
    });
    
    // Puzzle selection
    document.getElementById('puzzle-select').addEventListener('change', (e) => {
        state.currentTaskId = e.target.value;
        loadTask();
    });
    
    // Example number
    document.getElementById('example-num').addEventListener('change', (e) => {
        state.exampleNum = parseInt(e.target.value);
        loadExample();
    });
    
    // Search
    document.getElementById('search').addEventListener('input', (e) => {
        filterPuzzles(e.target.value);
    });
    
    // Navigation buttons
    document.getElementById('btn-first').addEventListener('click', () => navigate('first'));
    document.getElementById('btn-prev').addEventListener('click', () => navigate('prev'));
    document.getElementById('btn-next').addEventListener('click', () => navigate('next'));
    
    // Grid controls
    document.getElementById('btn-resize').addEventListener('click', () => resizeGrid());
    document.getElementById('btn-copy-input').addEventListener('click', () => copyFromInput());
    document.getElementById('btn-copy-output').addEventListener('click', () => copyFromOutput());
    document.getElementById('btn-copy-last-step').addEventListener('click', () => copyFromLastStep());
    document.getElementById('btn-clear').addEventListener('click', () => clearGrid());
    
    // Save button
    document.getElementById('btn-save').addEventListener('click', saveStep);
    
    // Last step toggle
    document.getElementById('toggle-last-step')?.addEventListener('click', toggleLastStep);
    
    // Drawing tools
    document.getElementById('tool-paint').addEventListener('click', () => setDrawingTool('paint'));
    document.getElementById('tool-fill').addEventListener('click', () => setDrawingTool('fill'));
    document.getElementById('tool-select').addEventListener('click', () => setDrawingTool('select'));
    document.getElementById('btn-fill-selection').addEventListener('click', fillSelection);
    
    // Edit mode
    document.getElementById('btn-edit-step').addEventListener('click', loadStepForEditing);
    document.getElementById('btn-cancel-edit').addEventListener('click', cancelEditMode);
    
    // AI rewrite
    document.getElementById('btn-ai-rewrite').addEventListener('click', aiRewriteDescription);
    
    // Step manager
    document.getElementById('btn-manage-steps').addEventListener('click', openStepManager);
    document.getElementById('btn-close-manager').addEventListener('click', closeStepManager);
    document.getElementById('btn-cancel-manager').addEventListener('click', closeStepManager);
    document.getElementById('btn-apply-changes').addEventListener('click', applyStepChanges);
}

// Step Manager
let stepManagerState = {
    steps: [],
    operations: []  // Track operations: {type: 'delete'|'move'|'insert', ...}
};

function setDrawingTool(tool) {
    state.drawingTool = tool;
    
    // Update button styles
    document.querySelectorAll('.tool-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`tool-${tool}`).classList.add('active');
    
    // Update info text
    const infoText = {
        paint: 'Click cells to paint with selected color',
        fill: 'Click and drag to select area, then release to fill',
        select: 'Click and drag to select cells, then click Fill button'
    };
    document.getElementById('tool-info').textContent = infoText[tool];
    
    // Show/hide fill button
    document.getElementById('btn-fill-selection').style.display = 
        tool === 'select' ? 'block' : 'none';
    
    // Clear any existing selection
    clearSelection();
}

// Load tasks from API
async function loadTasks() {
    try {
        const response = await fetch('/api/tasks');
        const data = await response.json();
        state.tasks = data;
        populatePuzzleSelect();
    } catch (error) {
        console.error('Error loading tasks:', error);
    }
}

function populatePuzzleSelect() {
    const select = document.getElementById('puzzle-select');
    const tasks = state.tasks[state.currentDataset];
    
    select.innerHTML = '';
    tasks.forEach(taskId => {
        const option = document.createElement('option');
        option.value = taskId;
        option.textContent = taskId;
        select.appendChild(option);
    });
    
    if (tasks.length > 0) {
        state.currentTaskId = tasks[0];
        select.value = state.currentTaskId;
        loadTask();
    }
}

function filterPuzzles(query) {
    const select = document.getElementById('puzzle-select');
    const tasks = state.tasks[state.currentDataset];
    const filtered = query 
        ? tasks.filter(t => t.toLowerCase().includes(query.toLowerCase()))
        : tasks;
    
    select.innerHTML = '';
    filtered.forEach(taskId => {
        const option = document.createElement('option');
        option.value = taskId;
        option.textContent = taskId;
        select.appendChild(option);
    });
}

function navigate(direction) {
    const select = document.getElementById('puzzle-select');
    const options = Array.from(select.options);
    const currentIndex = options.findIndex(opt => opt.value === state.currentTaskId);
    
    let newIndex;
    if (direction === 'first') newIndex = 0;
    else if (direction === 'prev') newIndex = Math.max(0, currentIndex - 1);
    else if (direction === 'next') newIndex = Math.min(options.length - 1, currentIndex + 1);
    
    if (newIndex !== undefined && options[newIndex]) {
        select.value = options[newIndex].value;
        state.currentTaskId = options[newIndex].value;
        loadTask();
    }
}

// Load task data
async function loadTask() {
    if (!state.currentTaskId) return;
    
    try {
        const response = await fetch(`/api/task/${state.currentDataset}/${state.currentTaskId}`);
        const data = await response.json();
        state.currentTaskData = data;
        
        // Load universal steps
        await loadUniversalSteps();
        
        // Update UI
        document.getElementById('current-puzzle-info').innerHTML = `
            <strong>Puzzle:</strong> <code>${state.currentTaskId}</code><br>
            <strong>Dataset:</strong> ${state.currentDataset}<br>
            <strong>Training:</strong> ${data.train.length} examples<br>
            <strong>Testing:</strong> ${data.test.length} examples<br>
            <strong>Universal Steps:</strong> ${state.universalSteps.length}
        `;
        
        populateExampleSelect();
        loadExample();
    } catch (error) {
        console.error('Error loading task:', error);
    }
}

async function loadUniversalSteps() {
    try {
        const response = await fetch(`/api/universal_steps/${state.currentTaskId}`);
        state.universalSteps = await response.json();
        updateUniversalStepsUI();
    } catch (error) {
        console.error('Error loading universal steps:', error);
        state.universalSteps = [];
        updateUniversalStepsUI();
    }
}

function updateUniversalStepsUI() {
    const list = document.getElementById('universal-steps-list');
    
    if (state.universalSteps.length === 0) {
        list.innerHTML = '<p style="color: #bdc3c7; font-size: 0.9rem;">No universal steps yet. Create steps on any example to build templates.</p>';
        return;
    }
    
    let html = '<div style="color: #3498db; font-weight: 600; margin-bottom: 0.5rem;">✨ ' + state.universalSteps.length + ' template(s) (most recent):</div>';
    state.universalSteps.forEach((step, idx) => {
        const isCurrent = idx === state.currentStepIndex;
        const bgColor = isCurrent ? 'rgba(52, 152, 219, 0.2)' : 'rgba(255,255,255,0.05)';
        const borderColor = isCurrent ? '#3498db' : 'rgba(52, 152, 219, 0.3)';
        
        const updatedFrom = step.updated_from ? `<br><small style="color: #95a5a6; font-size: 0.75rem;">Last from: ${step.updated_from}</small>` : '';
        
        html += `
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: ${bgColor}; 
                        border-radius: 3px; border-left: 3px solid ${borderColor}; font-size: 0.85rem;">
                <strong>Step ${idx + 1}${isCurrent ? ' ← Current' : ''}</strong><br>
                <span style="color: #ecf0f1;">${step.step_name || 'Unnamed'}</span>
                ${updatedFrom}
            </div>
        `;
    });
    
    list.innerHTML = html;
}

function populateExampleSelect() {
    if (!state.currentTaskData) return;
    
    const select = document.getElementById('example-num');
    const count = state.exampleType === 'training' 
        ? state.currentTaskData.train.length 
        : state.currentTaskData.test.length;
    
    select.innerHTML = '';
    for (let i = 1; i <= count; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = i;
        select.appendChild(option);
    }
    
    state.exampleNum = 1;
    select.value = 1;
}

// Load specific example
async function loadExample() {
    if (!state.currentTaskData) return;
    
    const examples = state.exampleType === 'training' 
        ? state.currentTaskData.train 
        : state.currentTaskData.test;
    
    const example = examples[state.exampleNum - 1];
    if (!example) return;
    
    // Update badge
    const badge = document.getElementById('example-badge');
    badge.textContent = `${state.exampleType.toUpperCase()} EXAMPLE ${String(state.exampleNum).padStart(2, '0')}`;
    badge.className = `example-badge ${state.exampleType === 'testing' ? 'testing' : ''}`;
    
    // Display original grids
    state.originalInput = example.input;
    state.originalOutput = example.output || null;
    
    renderGrid(example.input, 'original-input');
    document.getElementById('original-input-size').textContent = 
        `Size: ${example.input.length}×${example.input[0].length}`;
    
    if (state.originalOutput) {
        renderGrid(state.originalOutput, 'original-output');
        document.getElementById('original-output-size').textContent = 
            `Size: ${state.originalOutput.length}×${state.originalOutput[0].length}`;
    } else {
        document.getElementById('original-output').innerHTML = '<p style="color: #999;">No output available</p>';
        document.getElementById('original-output-size').textContent = '';
    }
    
    // Initialize editing grid - start with expected output if available, otherwise input
    state.grid = state.originalOutput 
        ? JSON.parse(JSON.stringify(state.originalOutput))
        : JSON.parse(JSON.stringify(example.input));
    
    // Update size inputs
    document.getElementById('grid-height').value = state.grid.length;
    document.getElementById('grid-width').value = state.grid[0].length;
    
    renderEditableGrid(state.grid, 'grid-editor');
    
    // Load steps and update UI
    loadSteps();
    
    // Load first universal step if available
    state.currentStepIndex = 0;
    loadCurrentStepTemplate();
}

// Load steps for current example
async function loadSteps() {
    try {
        const response = await fetch(
            `/api/steps/${state.currentTaskId}/${state.exampleType}/${state.exampleNum}`
        );
        const steps = await response.json();
        
        // Update steps info
        const stepsInfo = document.getElementById('steps-info');
        if (steps.length > 0) {
            let html = `<div style="color: #4caf50; font-weight: 600; margin-bottom: 0.5rem;">✅ ${steps.length} step(s) saved</div>`;
            steps.forEach((step, idx) => {
                html += `<div class="step-item step-clickable" data-step-index="${idx}" style="cursor: pointer;">
                    📄 Step ${String(step.step_num).padStart(2, '0')}: ${step.step_name || 'Unnamed'}
                </div>`;
            });
            stepsInfo.innerHTML = html;
            
            // Add click handlers to steps
            document.querySelectorAll('.step-clickable').forEach((el, idx) => {
                el.addEventListener('click', () => {
                    showStep(steps[idx]);
                });
            });
            
            // Show last step by default
            showStep(steps[steps.length - 1]);
            
            // Set current step index to next step
            state.currentStepIndex = steps.length;
            
            // Show manage steps button
            document.getElementById('btn-manage-steps').style.display = 'block';
        } else {
            stepsInfo.innerHTML = '<div style="color: #999;">📝 No steps yet</div>';
            hideLastStep();
            state.currentStepIndex = 0;
            
            // Hide manage steps button
            document.getElementById('btn-manage-steps').style.display = 'none';
        }
        
        // Update next location
        const nextStep = steps.length + 1;
        document.getElementById('next-location').textContent = 
            `visual_traces/\n  ${state.currentTaskId}/\n    ${state.exampleType}_${String(state.exampleNum).padStart(2, '0')}/\n      step_${String(nextStep).padStart(2, '0')}/`;
        
        // Update save info
        document.getElementById('save-info').innerHTML = 
            `📝 This will be <strong>Step ${String(nextStep).padStart(2, '0')}</strong> for <strong>${state.exampleType.charAt(0).toUpperCase() + state.exampleType.slice(1)} Example ${String(state.exampleNum).padStart(2, '0')}</strong>`;
        
        // Update step name default
        document.getElementById('step-name').value = `Step ${nextStep}`;
    } catch (error) {
        console.error('Error loading steps:', error);
    }
}

function loadCurrentStepTemplate() {
    // Load template from universal steps
    const currentStep = state.universalSteps[state.currentStepIndex];
    
    if (currentStep) {
        // Pre-fill form with universal step
        document.getElementById('step-name').value = currentStep.step_name || `Step ${state.currentStepIndex + 1}`;
        document.getElementById('step-description').value = currentStep.description || '';
        
        // Update UI to show it's from template
        document.getElementById('save-info').innerHTML = 
            `📝 <strong>Step ${String(state.currentStepIndex + 1).padStart(2, '0')}</strong> <span style="color: #3498db;">✨ (from universal template)</span><br><small>Edit as needed - changes will update the template</small>`;
    } else {
        // No template, clear form
        document.getElementById('step-name').value = `Step ${state.currentStepIndex + 1}`;
        document.getElementById('step-description').value = '';
        
        document.getElementById('save-info').innerHTML = 
            `📝 <strong>New Step ${String(state.currentStepIndex + 1).padStart(2, '0')}</strong> <span style="color: #4caf50;">✨ (creating new template)</span><br><small>This will be reused for other examples</small>`;
    }
    
    // Update universal steps UI to highlight current
    updateUniversalStepsUI();
}

function showStep(step) {
    const section = document.getElementById('last-step-section');
    section.style.display = 'block';
    
    // Update title to show which step is being viewed
    const stepNumStr = String(step.step_num).padStart(2, '0');
    document.getElementById('last-step-title').textContent = 
        `Step ${stepNumStr}: ${step.step_name || 'Unnamed'}`;
    
    renderGrid(step.grid, 'last-step-grid');
    
    // Show grid size
    if (step.grid && step.grid.length > 0) {
        document.getElementById('last-step-grid-size').textContent = 
            `Size: ${step.grid.length}×${step.grid[0].length}`;
    }
    
    document.getElementById('last-step-description').textContent = step.description || 'No description';
    
    // Show timestamp
    if (step.timestamp) {
        const timestamp = new Date(step.timestamp).toLocaleString();
        document.getElementById('last-step-timestamp').textContent = `Saved: ${timestamp}`;
        document.getElementById('last-step-timestamp').style.display = 'block';
    } else {
        document.getElementById('last-step-timestamp').style.display = 'none';
    }
    
    // Store step data for editing/copying
    state.lastStepGrid = step.grid;
    state.viewingStep = step;
    
    // Show both copy and edit buttons
    const copyBtn = document.getElementById('btn-copy-last-step');
    copyBtn.style.display = 'inline-block';
    copyBtn.textContent = `📋 Copy Step ${stepNumStr}`;
    
    const editBtn = document.getElementById('btn-edit-step');
    editBtn.style.display = 'block';
    editBtn.textContent = `✏️ Edit Step ${stepNumStr}`;
    
    // Highlight the clicked step in sidebar
    document.querySelectorAll('.step-clickable').forEach((el, idx) => {
        if (el.dataset.stepIndex == (step.step_num - 1)) {
            el.style.background = 'rgba(52, 152, 219, 0.3)';
            el.style.borderLeftColor = '#3498db';
            el.style.fontWeight = '600';
        } else {
            el.style.background = 'rgba(255,255,255,0.05)';
            el.style.borderLeftColor = 'var(--secondary-color)';
            el.style.fontWeight = 'normal';
        }
    });
    
    // Auto-expand the viewer
    const content = document.getElementById('last-step-content');
    const button = document.getElementById('toggle-last-step');
    content.style.display = 'block';
    button.classList.add('open');
}

function hideLastStep() {
    const section = document.getElementById('last-step-section');
    section.style.display = 'none';
    state.lastStepGrid = null;
    
    // Hide the "Copy Last Step" button
    document.getElementById('btn-copy-last-step').style.display = 'none';
    
    // Reset button text
    document.getElementById('btn-copy-last-step').textContent = '📋 Copy Last Step';
}

function toggleLastStep() {
    const content = document.getElementById('last-step-content');
    const button = document.getElementById('toggle-last-step');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        button.classList.add('open');
    } else {
        content.style.display = 'none';
        button.classList.remove('open');
    }
}

// Grid rendering
function renderGrid(grid, containerId) {
    const container = document.getElementById(containerId);
    const gridDiv = document.createElement('div');
    gridDiv.className = 'grid-display';
    
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

function renderEditableGrid(grid, containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    
    // Remove old global listener if exists
    document.removeEventListener('mouseup', handleGlobalMouseUp);
    
    grid.forEach((row, i) => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'grid-row-editor';
        
        row.forEach((value, j) => {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.style.backgroundColor = ARC_COLORS[value];
            cell.title = `(${i},${j}): ${COLOR_NAMES[value]}`;
            cell.dataset.row = i;
            cell.dataset.col = j;
            
            // Mouse down - start selection or paint
            cell.addEventListener('mousedown', (e) => {
                e.preventDefault();
                handleCellMouseDown(i, j, cell);
            });
            
            // Mouse enter - continue selection
            cell.addEventListener('mouseenter', () => {
                handleCellMouseEnter(i, j, cell);
            });
            
            // Mouse up - end selection
            cell.addEventListener('mouseup', () => {
                handleCellMouseUp(i, j);
            });
            
            rowDiv.appendChild(cell);
        });
        
        container.appendChild(rowDiv);
    });
    
    // Add global mouse up handler (only once)
    document.addEventListener('mouseup', handleGlobalMouseUp);
}

function handleCellMouseDown(row, col, cell) {
    state.isSelecting = true;
    
    if (state.drawingTool === 'paint') {
        // Paint mode - immediate paint
        state.grid[row][col] = state.selectedColor;
        cell.style.backgroundColor = ARC_COLORS[state.selectedColor];
        cell.title = `(${row},${col}): ${COLOR_NAMES[state.selectedColor]}`;
    } else if (state.drawingTool === 'fill' || state.drawingTool === 'select') {
        // Selection modes - start selection
        state.selectionStart = {row, col};
        state.selectionEnd = {row, col};
        clearSelection();
        updateSelection();
    }
}

function handleCellMouseEnter(row, col, cell) {
    if (state.drawingTool === 'paint' && state.isSelecting) {
        // Paint mode with mouse down - paint
        state.grid[row][col] = state.selectedColor;
        cell.style.backgroundColor = ARC_COLORS[state.selectedColor];
        cell.title = `(${row},${col}): ${COLOR_NAMES[state.selectedColor]}`;
    } else if ((state.drawingTool === 'fill' || state.drawingTool === 'select') && state.isSelecting) {
        // Selection modes - update selection
        state.selectionEnd = {row, col};
        updateSelection();
    }
}

function handleCellMouseUp(row, col) {
    if (!state.isSelecting) return;
    
    if (state.drawingTool === 'fill') {
        // Fill mode - fill and clear
        fillSelection();
        clearSelection();
    } else if (state.drawingTool === 'select') {
        // Select mode - keep selection visible
        // Don't clear, user will click fill button
    }
    
    state.isSelecting = false;
}

function handleGlobalMouseUp() {
    if (!state.isSelecting) return;
    
    if (state.drawingTool === 'fill') {
        // Fill mode - fill and clear even if released outside grid
        fillSelection();
        clearSelection();
    } else if (state.drawingTool === 'select') {
        // Select mode - keep selection visible
    }
    
    state.isSelecting = false;
}

function updateSelection() {
    if (!state.selectionStart || !state.selectionEnd) return;
    
    // Clear previous selection visuals
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.classList.remove('selecting', 'selected');
    });
    
    // Calculate bounds
    const minRow = Math.min(state.selectionStart.row, state.selectionEnd.row);
    const maxRow = Math.max(state.selectionStart.row, state.selectionEnd.row);
    const minCol = Math.min(state.selectionStart.col, state.selectionEnd.col);
    const maxCol = Math.max(state.selectionStart.col, state.selectionEnd.col);
    
    // Update selected cells
    state.selectedCells.clear();
    document.querySelectorAll('.grid-cell').forEach(cell => {
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        
        if (row >= minRow && row <= maxRow && col >= minCol && col <= maxCol) {
            if (state.isSelecting) {
                cell.classList.add('selecting');
            } else {
                cell.classList.add('selected');
            }
            state.selectedCells.add(`${row},${col}`);
        }
    });
}

function fillSelection() {
    if (state.selectedCells.size === 0) {
        console.log('No cells selected');
        return;
    }
    
    console.log(`Filling ${state.selectedCells.size} cells with color ${state.selectedColor}`);
    
    state.selectedCells.forEach(cellKey => {
        const [row, col] = cellKey.split(',').map(Number);
        state.grid[row][col] = state.selectedColor;
        
        // Update visual
        const cell = document.querySelector(`.grid-cell[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            cell.style.backgroundColor = ARC_COLORS[state.selectedColor];
            cell.title = `(${row},${col}): ${COLOR_NAMES[state.selectedColor]}`;
            cell.classList.remove('selecting', 'selected');
        }
    });
    
    if (state.drawingTool === 'select') {
        clearSelection();
    }
}

function clearSelection() {
    state.selectedCells.clear();
    state.selectionStart = null;
    state.selectionEnd = null;
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.classList.remove('selecting', 'selected');
    });
}

// Color palette
function initializeColorPalette() {
    const palette = document.getElementById('color-palette');
    
    for (let i = 0; i < 10; i++) {
        const button = document.createElement('div');
        button.className = 'color-button';
        if (i === state.selectedColor) button.classList.add('selected');
        button.style.backgroundColor = ARC_COLORS[i];
        button.title = `${i}: ${COLOR_NAMES[i]}`;
        
        const number = document.createElement('div');
        number.className = 'color-number';
        number.textContent = i;
        button.appendChild(number);
        
        button.addEventListener('click', () => {
            state.selectedColor = i;
            updateColorSelection();
        });
        
        palette.appendChild(button);
    }
    
    updateColorSelection();
}

function updateColorSelection() {
    document.querySelectorAll('.color-button').forEach((btn, i) => {
        if (i === state.selectedColor) {
            btn.classList.add('selected');
        } else {
            btn.classList.remove('selected');
        }
    });
    
    document.getElementById('selected-color-info').textContent = 
        `Selected: ${COLOR_NAMES[state.selectedColor]} (${state.selectedColor})`;
}

// Grid operations
function resizeGrid() {
    const height = parseInt(document.getElementById('grid-height').value);
    const width = parseInt(document.getElementById('grid-width').value);
    
    state.grid = Array(height).fill(0).map(() => Array(width).fill(0));
    clearSelection();
    renderEditableGrid(state.grid, 'grid-editor');
}

function copyFromInput() {
    state.grid = JSON.parse(JSON.stringify(state.originalInput));
    document.getElementById('grid-height').value = state.grid.length;
    document.getElementById('grid-width').value = state.grid[0].length;
    clearSelection();
    renderEditableGrid(state.grid, 'grid-editor');
}

function copyFromOutput() {
    if (state.originalOutput) {
        state.grid = JSON.parse(JSON.stringify(state.originalOutput));
        document.getElementById('grid-height').value = state.grid.length;
        document.getElementById('grid-width').value = state.grid[0].length;
        clearSelection();
        renderEditableGrid(state.grid, 'grid-editor');
    } else {
        alert('No expected output available for this example');
    }
}

function copyFromLastStep() {
    if (state.lastStepGrid) {
        state.grid = JSON.parse(JSON.stringify(state.lastStepGrid));
        document.getElementById('grid-height').value = state.grid.length;
        document.getElementById('grid-width').value = state.grid[0].length;
        clearSelection();
        renderEditableGrid(state.grid, 'grid-editor');
        
        showSuccessMessage('✅ Copied grid from step!');
    } else {
        alert('No step available to copy from');
    }
}

function loadStepForEditing() {
    if (!state.viewingStep) {
        alert('No step selected to edit');
        return;
    }
    
    // Enter edit mode
    state.editMode = true;
    state.editingStep = state.viewingStep;
    
    // Load step data into form
    document.getElementById('step-name').value = state.editingStep.step_name || '';
    document.getElementById('step-description').value = state.editingStep.description || '';
    
    // Load grid
    state.grid = JSON.parse(JSON.stringify(state.editingStep.grid));
    document.getElementById('grid-height').value = state.grid.length;
    document.getElementById('grid-width').value = state.grid[0].length;
    clearSelection();
    renderEditableGrid(state.grid, 'grid-editor');
    
    // Update UI
    updateEditModeUI();
    
    // Scroll to editor
    document.querySelector('.section-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    showSuccessMessage(`✏️ Loaded Step ${String(state.editingStep.step_num).padStart(2, '0')} for editing`);
}

function cancelEditMode() {
    state.editMode = false;
    state.editingStep = null;
    
    // Reset to new step mode
    state.currentStepIndex = state.universalSteps.length;
    loadCurrentStepTemplate();
    
    // Reset grid
    state.grid = state.originalOutput 
        ? JSON.parse(JSON.stringify(state.originalOutput))
        : JSON.parse(JSON.stringify(state.originalInput));
    renderEditableGrid(state.grid, 'grid-editor');
    
    updateEditModeUI();
    
    showSuccessMessage('✅ Canceled edit mode - back to creating new step');
}

function updateEditModeUI() {
    const editBanner = document.getElementById('edit-mode-banner');
    const saveBtn = document.getElementById('btn-save');
    const saveInfo = document.getElementById('save-info');
    
    if (state.editMode && state.editingStep) {
        // Show edit mode banner
        editBanner.style.display = 'flex';
        
        // Update save button
        const stepNumStr = String(state.editingStep.step_num).padStart(2, '0');
        saveBtn.textContent = `💾 Update Step ${stepNumStr}`;
        saveBtn.style.background = '#ff9800';
        
        // Update save info
        saveInfo.innerHTML = `✏️ <strong>Editing Step ${stepNumStr}</strong> - Changes will update this step and the universal template`;
        saveInfo.style.background = '#fff3cd';
        saveInfo.style.borderLeftColor = '#ff9800';
    } else {
        // Hide edit mode banner
        editBanner.style.display = 'none';
        
        // Reset save button
        saveBtn.textContent = '💾 Save Step';
        saveBtn.style.background = 'var(--success-color)';
        
        // Normal save info
        const nextStep = state.currentStepIndex + 1;
        loadCurrentStepTemplate();
        saveInfo.style.background = '#e3f2fd';
        saveInfo.style.borderLeftColor = '#2196f3';
    }
}

function clearGrid() {
    state.grid = state.grid.map(row => row.map(() => 0));
    clearSelection();
    renderEditableGrid(state.grid, 'grid-editor');
}

// Save step
async function saveStep() {
    const description = document.getElementById('step-description').value.trim();
    
    if (!description) {
        alert('Please add a description for this step');
        return;
    }
    
    let stepNum;
    let isUpdate = false;
    
    if (state.editMode && state.editingStep) {
        // Editing existing step
        stepNum = state.editingStep.step_num;
        isUpdate = true;
    } else {
        // Creating new step
        stepNum = state.currentStepIndex + 1;
    }
    
    const stepData = {
        task_id: state.currentTaskId,
        example_type: state.exampleType,
        example_num: state.exampleNum,
        step_num: stepNum,
        step_name: document.getElementById('step-name').value || `Step ${stepNum}`,
        description: description,
        grid: state.grid,
        original_input: state.originalInput,
        original_output: state.originalOutput
    };
    
    try {
        const response = await fetch('/api/save_step', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(stepData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            if (isUpdate) {
                showSuccessMessage(`✅ Step ${String(stepNum).padStart(2, '0')} updated successfully!`);
                
                // Exit edit mode
                state.editMode = false;
                state.editingStep = null;
                updateEditModeUI();
            } else {
                showSuccessMessage(`✅ ${result.message} & updated universal template`);
                
                // Move to next step
                state.currentStepIndex++;
            }
            
            // Reload universal steps
            await loadUniversalSteps();
            
            // Load next template (unless we just updated)
            if (!isUpdate) {
                loadCurrentStepTemplate();
            }
            
            // Reload example steps
            await loadSteps();
            
            // Clear grid for next step (unless editing - then reload the updated step)
            if (!isUpdate) {
                state.grid = state.originalOutput 
                    ? JSON.parse(JSON.stringify(state.originalOutput))
                    : JSON.parse(JSON.stringify(state.originalInput));
                renderEditableGrid(state.grid, 'grid-editor');
            }
        } else {
            alert('Error saving step');
        }
    } catch (error) {
        console.error('Error saving step:', error);
        alert('Error saving step');
    }
}

function showSuccessMessage(message) {
    const msgDiv = document.getElementById('success-message');
    msgDiv.textContent = message;
    msgDiv.style.display = 'block';
    
    setTimeout(() => {
        msgDiv.style.display = 'none';
    }, 3000);
}

// AI Rewrite feature
async function aiRewriteDescription() {
    const description = document.getElementById('step-description').value.trim();
    
    if (!description) {
        alert('Please write a description first, then AI can rewrite it to be clearer');
        return;
    }
    
    const button = document.getElementById('btn-ai-rewrite');
    const statusDiv = document.getElementById('ai-rewrite-status');
    
    // Figure out current step number
    let currentStepNum;
    if (state.editMode && state.editingStep) {
        currentStepNum = state.editingStep.step_num;
    } else {
        currentStepNum = state.currentStepIndex + 1;
    }
    
    // Disable button and show loading
    button.disabled = true;
    button.textContent = '🤖 Rewriting...';
    statusDiv.className = 'ai-rewrite-status loading';
    
    // Show context info
    const contextCount = state.universalSteps.filter(s => s.step_num < currentStepNum).length;
    if (contextCount > 0) {
        statusDiv.textContent = `⏳ AI is analyzing ${contextCount} previous step(s) and rewriting...`;
    } else {
        statusDiv.textContent = '⏳ AI is rewriting your description...';
    }
    statusDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/ai_rewrite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                description: description,
                task_id: state.currentTaskId,
                step_num: currentStepNum,
                example_type: state.exampleType,
                example_num: state.exampleNum,
                step_name: document.getElementById('step-name').value
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Update textarea with rewritten description
            document.getElementById('step-description').value = result.rewritten;
            
            // Show success with context info
            statusDiv.className = 'ai-rewrite-status success';
            let successMsg = `✅ <strong>Rewritten!</strong> ${description.length} chars → ${result.rewritten.length} chars`;
            if (result.context_steps > 0) {
                successMsg += ` (used ${result.context_steps} previous step(s) for context)`;
            }
            statusDiv.innerHTML = successMsg;
            
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        } else {
            // Show error
            statusDiv.className = 'ai-rewrite-status error';
            statusDiv.textContent = `❌ ${result.error || 'Error rewriting description'}`;
        }
    } catch (error) {
        console.error('Error rewriting:', error);
        statusDiv.className = 'ai-rewrite-status error';
        statusDiv.textContent = '❌ Error connecting to AI service';
    } finally {
        // Re-enable button
        button.disabled = false;
        button.textContent = '🤖 AI Rewrite (2-3 sentences)';
    }
}

function openStepManager() {
    // Load current steps
    fetch(`/api/steps/${state.currentTaskId}/${state.exampleType}/${state.exampleNum}`)
        .then(res => res.json())
        .then(steps => {
            stepManagerState.steps = steps.map(s => ({...s, markedForDeletion: false}));
            stepManagerState.operations = [];
            renderStepManager();
            document.getElementById('step-manager-modal').style.display = 'flex';
        });
}

function closeStepManager() {
    document.getElementById('step-manager-modal').style.display = 'none';
    stepManagerState = {steps: [], operations: []};
}

function renderStepManager() {
    const list = document.getElementById('step-manager-list');
    
    if (stepManagerState.steps.length === 0) {
        list.innerHTML = '<p style="color: #999;">No steps to manage</p>';
        return;
    }
    
    let html = '';
    stepManagerState.steps.forEach((step, idx) => {
        const willBeNum = idx + 1;
        const isFirst = idx === 0;
        const isLast = idx === stepManagerState.steps.length - 1;
        const itemClass = step.markedForDeletion ? 'step-manager-item will-delete' : 'step-manager-item';
        
        html += `
            <div class="${itemClass}">
                <div class="step-num-badge">
                    ${step.markedForDeletion ? '🗑️' : `Step ${String(willBeNum).padStart(2, '0')}`}
                </div>
                <div class="step-info">
                    <div class="step-info-name">${step.step_name || 'Unnamed'}</div>
                    <div class="step-info-desc">${step.description || 'No description'}</div>
                    ${step.step_num !== willBeNum && !step.markedForDeletion ? `<small style="color: #ff9800;">Will renumber: ${step.step_num} → ${willBeNum}</small>` : ''}
                </div>
                <div class="step-actions-manager">
                    <button class="btn-step-manager" onclick="moveStepUp(${idx})" ${isFirst ? 'disabled' : ''} title="Move up">
                        ▲
                    </button>
                    <button class="btn-step-manager" onclick="moveStepDown(${idx})" ${isLast ? 'disabled' : ''} title="Move down">
                        ▼
                    </button>
                    <button class="btn-step-manager btn-insert" onclick="insertStepAfter(${idx})" title="Insert step after this">
                        ➕
                    </button>
                    <button class="btn-step-manager btn-delete" onclick="toggleDelete(${idx})" title="${step.markedForDeletion ? 'Undo delete' : 'Delete step'}">
                        ${step.markedForDeletion ? '↩️' : '🗑️'}
                    </button>
                </div>
            </div>
        `;
    });
    
    list.innerHTML = html;
}

function moveStepUp(idx) {
    if (idx === 0) return;
    const temp = stepManagerState.steps[idx];
    stepManagerState.steps[idx] = stepManagerState.steps[idx - 1];
    stepManagerState.steps[idx - 1] = temp;
    renderStepManager();
}

function moveStepDown(idx) {
    if (idx === stepManagerState.steps.length - 1) return;
    const temp = stepManagerState.steps[idx];
    stepManagerState.steps[idx] = stepManagerState.steps[idx + 1];
    stepManagerState.steps[idx + 1] = temp;
    renderStepManager();
}

function toggleDelete(idx) {
    stepManagerState.steps[idx].markedForDeletion = !stepManagerState.steps[idx].markedForDeletion;
    renderStepManager();
}

function insertStepAfter(idx) {
    const newStep = {
        step_num: -1,  // Will be assigned
        step_name: `New Step ${idx + 2}`,
        description: '',
        grid: null,
        isNew: true,
        markedForDeletion: false
    };
    stepManagerState.steps.splice(idx + 1, 0, newStep);
    renderStepManager();
}

async function applyStepChanges() {
    const button = document.getElementById('btn-apply-changes');
    button.disabled = true;
    button.textContent = '⏳ Applying changes...';
    
    try {
        // Build reorder data
        const reorderData = {
            task_id: state.currentTaskId,
            example_type: state.exampleType,
            example_num: state.exampleNum,
            steps: stepManagerState.steps.map((step, idx) => ({
                old_step_num: step.step_num,
                new_step_num: idx + 1,
                markedForDeletion: step.markedForDeletion,
                isNew: step.isNew || false,
                step_name: step.step_name,
                description: step.description
            }))
        };
        
        const response = await fetch('/api/reorder_steps', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(reorderData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showSuccessMessage(`✅ ${result.message}`);
            closeStepManager();
            
            // Reload steps
            await loadUniversalSteps();
            await loadSteps();
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error('Error applying changes:', error);
        alert('Error applying changes');
    } finally {
        button.disabled = false;
        button.textContent = '💾 Apply Changes';
    }
}

