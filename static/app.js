/* ==========================================================================
   DigitVision AI - Frontend Logic
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");
    const brushSizeSlider = document.getElementById("brush-size");
    const brushSizeVal = document.getElementById("brush-size-val");
    const clearBtn = document.getElementById("clear-btn");
    const predictBtn = document.getElementById("predict-btn");
    const realtimeToggle = document.getElementById("realtime-toggle");
    
    const digitDisplay = document.getElementById("digit-display");
    const confidencePct = document.getElementById("confidence-percentage");
    const predictedValue = document.getElementById("predicted-value");
    const probListContainer = document.querySelector(".probability-list");
    const probRowTemplate = document.getElementById("prob-row-template");
    
    const fmapL1Grid = document.getElementById("fmap-l1-grid");
    const fmapL2Grid = document.getElementById("fmap-l2-grid");
    
    const modelStatusBadge = document.getElementById("model-status-badge");
    const trainStatusBadge = document.getElementById("train-status-badge");
    
    // Hyperparameters
    const paramOptimizer = document.getElementById("param-optimizer");
    const paramLr = document.getElementById("param-lr");
    const paramComplexity = document.getElementById("param-complexity");
    const paramSize = document.getElementById("param-size");
    const paramEpochs = document.getElementById("param-epochs");
    const paramBatch = document.getElementById("param-batch");
    
    const startTrainBtn = document.getElementById("start-train-btn");
    const stopTrainBtn = document.getElementById("stop-train-btn");
    const progressContainer = document.getElementById("progress-container");
    const progressMessage = document.getElementById("progress-message");
    const epochCounter = document.getElementById("epoch-counter");
    const trainingProgressBar = document.getElementById("training-progress-bar");
    
    // ----------------------------------------------------
    // State Variables
    // ----------------------------------------------------
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let brushSize = parseInt(brushSizeSlider.value);
    
    let chart = null;
    let trainingPollInterval = null;
    let isTrainingActive = false;

    // ----------------------------------------------------
    // Initialize Draw Canvas
    // ----------------------------------------------------
    function initCanvas() {
        // Fill canvas with black background (MNIST compatibility: white strokes on black)
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Brush settings
        ctx.strokeStyle = "#ffffff";
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = brushSize;

        // Mouse Listeners
        canvas.addEventListener("mousedown", startDraw);
        canvas.addEventListener("mousemove", draw);
        canvas.addEventListener("mouseup", stopDraw);
        canvas.addEventListener("mouseleave", stopDraw);

        // Touch Listeners (Mobile support)
        canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            startDraw({
                clientX: touch.clientX,
                clientY: touch.clientY,
                target: canvas
            });
        });
        
        canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            draw({
                clientX: touch.clientX,
                clientY: touch.clientY
            });
        });
        
        canvas.addEventListener("touchend", (e) => {
            e.preventDefault();
            stopDraw();
        });
    }

    function startDraw(e) {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        // Calculate coords relative to canvas scaling
        lastX = ((e.clientX - rect.left) / rect.width) * canvas.width;
        lastY = ((e.clientY - rect.top) / rect.height) * canvas.height;
    }

    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const currX = ((e.clientX - rect.left) / rect.width) * canvas.width;
        const currY = ((e.clientY - rect.top) / rect.height) * canvas.height;
        
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currX, currY);
        ctx.stroke();
        
        lastX = currX;
        lastY = currY;
    }

    function stopDraw() {
        if (isDrawing) {
            isDrawing = false;
            // Real-time prediction trigger
            if (realtimeToggle.checked) {
                runInference();
            }
        }
    }

    // Brush controls
    brushSizeSlider.addEventListener("input", (e) => {
        brushSize = parseInt(e.target.value);
        brushSizeVal.textContent = brushSize + "px";
        ctx.lineWidth = brushSize;
    });

    clearBtn.addEventListener("click", () => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resetPredictions();
        resetFeatureMaps();
    });

    // ----------------------------------------------------
    // Inference Management
    // ----------------------------------------------------
    async function runInference() {
        // Get canvas data as PNG Base64
        const imgData = canvas.toDataURL("image/png");
        
        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: imgData })
            });
            const data = await response.json();
            
            if (data.success) {
                updatePredictionUI(data);
            } else {
                console.error("Prediction Error:", data.error);
            }
        } catch (error) {
            console.error("Failed to run inference API:", error);
        }
    }

    predictBtn.addEventListener("click", runInference);

    function updatePredictionUI(data) {
        const probs = data.predictions;
        const topDigit = data.top_digit;
        const confidence = probs[topDigit];
        
        // Update top display
        digitDisplay.textContent = topDigit;
        confidencePct.textContent = Math.round(confidence * 100) + "%";
        predictedValue.textContent = `${topDigit} (${Math.round(confidence * 100)}% Confidence)`;
        
        // Re-render Probability List
        probListContainer.innerHTML = "";
        
        probs.forEach((prob, idx) => {
            const clone = probRowTemplate.content.cloneNode(true);
            const row = clone.querySelector(".prob-row");
            const label = clone.querySelector(".digit-label");
            const fill = clone.querySelector(".progress-bar-fill");
            const pct = clone.querySelector(".digit-pct");
            
            label.textContent = idx;
            const pctVal = Math.round(prob * 100);
            fill.style.width = pctVal + "%";
            pct.textContent = pctVal + "%";
            
            if (idx === topDigit) {
                row.classList.add("active-row");
            }
            
            probListContainer.appendChild(clone);
        });
        
        // Render Feature Maps
        renderFeatureMaps(data.feature_maps_l1, data.feature_maps_l2);
    }

    function renderFeatureMaps(l1Maps, l2Maps) {
        // Render Layer 1 maps
        fmapL1Grid.innerHTML = "";
        if (l1Maps && l1Maps.length > 0) {
            l1Maps.forEach(mapSrc => {
                const item = document.createElement("div");
                item.className = "fmap-item";
                const img = document.createElement("img");
                img.src = mapSrc;
                img.alt = "Conv1 Activation Map";
                item.appendChild(img);
                fmapL1Grid.appendChild(item);
            });
        } else {
            fmapL1Grid.innerHTML = `<div class="fmap-placeholder">Error generating maps</div>`;
        }

        // Render Layer 2 maps
        fmapL2Grid.innerHTML = "";
        if (l2Maps && l2Maps.length > 0) {
            l2Maps.forEach(mapSrc => {
                const item = document.createElement("div");
                item.className = "fmap-item";
                const img = document.createElement("img");
                img.src = mapSrc;
                img.alt = "Conv2 Activation Map";
                item.appendChild(img);
                fmapL2Grid.appendChild(item);
            });
        } else {
            fmapL2Grid.innerHTML = `<div class="fmap-placeholder">Error generating maps</div>`;
        }
    }

    function resetPredictions() {
        digitDisplay.textContent = "?";
        confidencePct.textContent = "0%";
        predictedValue.textContent = "-";
        
        probListContainer.innerHTML = "";
        for (let i = 0; i < 10; i++) {
            const clone = probRowTemplate.content.cloneNode(true);
            const label = clone.querySelector(".digit-label");
            const fill = clone.querySelector(".progress-bar-fill");
            const pct = clone.querySelector(".digit-pct");
            label.textContent = i;
            fill.style.width = "0%";
            pct.textContent = "0%";
            probListContainer.appendChild(clone);
        }
    }

    function resetFeatureMaps() {
        fmapL1Grid.innerHTML = `<div class="fmap-placeholder">Draw on canvas to generate maps...</div>`;
        fmapL2Grid.innerHTML = `<div class="fmap-placeholder">Draw on canvas to generate maps...</div>`;
    }

    // ----------------------------------------------------
    // Hyperparameter Training Operations
    // ----------------------------------------------------
    async function startTraining() {
        const payload = {
            optimizer: paramOptimizer.value,
            lr: parseFloat(paramLr.value),
            complexity: paramComplexity.value,
            train_size: parseInt(paramSize.value),
            epochs: parseInt(paramEpochs.value),
            batch_size: parseInt(paramBatch.value)
        };
        
        try {
            const response = await fetch("/api/train/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            
            if (data.success) {
                setTrainingUIState(true);
                // Reset metrics chart
                resetChart();
                // Start polling status
                pollTrainingStatus();
            } else {
                alert("Error starting training: " + data.message);
            }
        } catch (error) {
            console.error("Failed to start training:", error);
        }
    }

    async function stopTraining() {
        try {
            await fetch("/api/train/stop", { method: "POST" });
        } catch (error) {
            console.error("Failed to submit stop request:", error);
        }
    }

    function setTrainingUIState(active) {
        isTrainingActive = active;
        startTrainBtn.disabled = active;
        stopTrainBtn.disabled = !active;
        
        // Lock/Unlock inputs
        paramOptimizer.disabled = active;
        paramLr.disabled = active;
        paramComplexity.disabled = active;
        paramSize.disabled = active;
        paramEpochs.disabled = active;
        paramBatch.disabled = active;
        
        if (active) {
            progressContainer.style.display = "block";
            trainStatusBadge.textContent = "Training";
            trainStatusBadge.className = "badge training-badge";
            modelStatusBadge.textContent = "Updating model...";
            modelStatusBadge.className = "badge";
        } else {
            trainStatusBadge.textContent = "Idle";
            trainStatusBadge.className = "badge";
        }
    }

    function pollTrainingStatus() {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        
        trainingPollInterval = setInterval(async () => {
            try {
                const response = await fetch("/api/train/status");
                const data = await response.json();
                
                // Update progress elements
                progressMessage.textContent = data.status_message;
                epochCounter.textContent = `Epoch ${data.current_epoch}/${data.total_epochs}`;
                
                if (data.total_epochs > 0) {
                    const progressPercent = (data.current_epoch / data.total_epochs) * 100;
                    trainingProgressBar.style.width = progressPercent + "%";
                }
                
                // Update live chart values
                if (data.loss_history && data.loss_history.length > 0) {
                    updateChartData(data.loss_history, data.accuracy_history);
                }
                
                // Check if finished
                if (!data.is_training) {
                    clearInterval(trainingPollInterval);
                    setTrainingUIState(false);
                    modelStatusBadge.textContent = "Model Ready";
                    modelStatusBadge.className = "badge active-badge";
                    
                    // Final status alert message
                    progressMessage.textContent = data.status_message;
                    trainingProgressBar.style.width = "100%";
                    
                    // Trigger a clean inference redraw on completion to update mapping
                    if (realtimeToggle.checked) {
                        runInference();
                    }
                }
            } catch (error) {
                console.error("Error polling training status:", error);
            }
        }, 1000);
    }

    startTrainBtn.addEventListener("click", startTraining);
    stopTrainBtn.addEventListener("click", stopTraining);

    // ----------------------------------------------------
    // Chart.js Configuration
    // ----------------------------------------------------
    function initChart() {
        const chartCtx = document.getElementById("metrics-chart").getContext("2d");
        
        chart = new Chart(chartCtx, {
            type: 'line',
            data: {
                labels: [], // Epoch numbers
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.05)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y-loss',
                    },
                    {
                        label: 'Validation Accuracy (%)',
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.05)',
                        borderWidth: 2,
                        tension: 0.3,
                        yAxisID: 'y-acc',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Inter', size: 11 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'JetBrains Mono' } },
                        title: { display: true, text: 'Epoch', color: '#9ca3af' }
                    },
                    'y-loss': {
                        type: 'linear',
                        position: 'left',
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#06b6d4', font: { family: 'JetBrains Mono' } },
                        title: { display: true, text: 'Loss', color: '#06b6d4' }
                    },
                    'y-acc': {
                        type: 'linear',
                        position: 'right',
                        min: 0,
                        max: 100,
                        grid: { drawOnChartArea: false }, // Avoid duplicate lines
                        ticks: { color: '#a855f7', font: { family: 'JetBrains Mono' } },
                        title: { display: true, text: 'Accuracy (%)', color: '#a855f7' }
                    }
                }
            }
        });
    }

    function resetChart() {
        if (!chart) return;
        chart.data.labels = [];
        chart.data.datasets[0].data = [];
        chart.data.datasets[1].data = [];
        chart.update();
    }

    function updateChartData(losses, accs) {
        if (!chart) return;
        
        // Build labels (Epoch 1, Epoch 2...)
        const labels = losses.map((_, idx) => `E${idx + 1}`);
        
        chart.data.labels = labels;
        chart.data.datasets[0].data = losses;
        chart.data.datasets[1].data = accs;
        chart.update();
    }

    // ----------------------------------------------------
    // Startup Sequence
    // ----------------------------------------------------
    async function checkServerStatus() {
        try {
            // Check if model weights are present on startup
            const response = await fetch("/api/model/info");
            const info = await response.json();
            
            // Check if backend is currently auto-training
            const statusResponse = await fetch("/api/train/status");
            const status = await statusResponse.json();
            
            if (status.is_training) {
                setTrainingUIState(true);
                pollTrainingStatus();
            } else if (info.has_weights) {
                modelStatusBadge.textContent = "Model Ready";
                modelStatusBadge.className = "badge active-badge";
            } else {
                modelStatusBadge.textContent = "Untrained";
                modelStatusBadge.className = "badge";
            }
        } catch (error) {
            console.error("Could not fetch server startup status:", error);
        }
    }

    // Initialize Page features
    initCanvas();
    resetPredictions();
    initChart();
    checkServerStatus();
});
