/* ----------------------------------------------------
   SpamGuard AI: Interactive Dynamic Logic
   ---------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------
    // 1. STATE VARIABLES & CONFIGS
    // ------------------------------------------------
    let activeModelTab = "naive_bayes";
    let isPollingStatus = false;
    let pollIntervalId = null;
    let samplesCached = [];
    
    // UI Elements
    const scanInput = document.getElementById("scanInput");
    const charCount = document.getElementById("charCount");
    const clearBtn = document.getElementById("clearBtn");
    const scanBtn = document.getElementById("scanBtn");
    const presetPills = document.getElementById("presetPills");
    
    // Model Selection for scanning
    const btnPredNB = document.getElementById("btnPredNB");
    const btnPredSVM = document.getElementById("btnPredSVM");
    let activePredModel = "naive_bayes";

    // Results Card
    const resultsCard = document.getElementById("resultsCard");
    const closeResultsBtn = document.getElementById("closeResultsBtn");
    const resultRing = document.getElementById("resultRing");
    const resultPercentage = document.getElementById("resultPercentage");
    const resultBadge = document.getElementById("resultBadge");
    const resultModelName = document.getElementById("resultModelName");
    const resultConfidence = document.getElementById("resultConfidence");
    const resultWordCount = document.getElementById("resultWordCount");
    const highlightedCanvas = document.getElementById("highlightedCanvas");

    // Hyperparameter Tabs
    const tabNB = document.getElementById("tabNB");
    const tabSVM = document.getElementById("tabSVM");
    const nbParamsSec = document.getElementById("nbParamsSec");
    const svmParamsSec = document.getElementById("svmParamsSec");
    
    // Hyperparameter Values
    const maxFeatures = document.getElementById("maxFeatures");
    const stopWords = document.getElementById("stopWords");
    const ngramRange = document.getElementById("ngramRange");
    const splitRatio = document.getElementById("splitRatio");
    const splitRatioVal = document.getElementById("splitRatioVal");
    
    const nbAlpha = document.getElementById("nbAlpha");
    const nbAlphaVal = document.getElementById("nbAlphaVal");
    
    const svmC = document.getElementById("svmC");
    const svmCVal = document.getElementById("svmCVal");
    const svmKernel = document.getElementById("svmKernel");
    
    const subsetSize = document.getElementById("subsetSize");
    const subsetSizeVal = document.getElementById("subsetSizeVal");
    
    // Training Button
    const trainBtn = document.getElementById("trainBtn");
    const trainBtnIcon = document.getElementById("trainBtnIcon");
    const statusMessageBadge = document.getElementById("statusMessageBadge");
    const consoleLogs = document.getElementById("consoleLogs");

    // Comparison Metrics (Naive Bayes)
    const accNB = document.getElementById("accNB");
    const accBarNB = document.getElementById("accBarNB");
    const precNB = document.getElementById("precNB");
    const precBarNB = document.getElementById("precBarNB");
    const recNB = document.getElementById("recNB");
    const recBarNB = document.getElementById("recBarNB");
    const f1NB = document.getElementById("f1NB");
    const f1BarNB = document.getElementById("f1BarNB");
    const trainedDotNB = document.getElementById("trainedDotNB");
    const statColNB = document.getElementById("statColNB");

    // Comparison Metrics (SVM)
    const accSVM = document.getElementById("accSVM");
    const accBarSVM = document.getElementById("accBarSVM");
    const precSVM = document.getElementById("precSVM");
    const precBarSVM = document.getElementById("precBarSVM");
    const recSVM = document.getElementById("recSVM");
    const recBarSVM = document.getElementById("recBarSVM");
    const f1SVM = document.getElementById("f1SVM");
    const f1BarSVM = document.getElementById("f1BarSVM");
    const trainedDotSVM = document.getElementById("trainedDotSVM");
    const statColSVM = document.getElementById("statColSVM");

    // Confusion Matrix Elements
    const cmModelName = document.getElementById("cmModelName");
    const cmTN = document.getElementById("cmTN");
    const cmFP = document.getElementById("cmFP");
    const cmFN = document.getElementById("cmFN");
    const cmTP = document.getElementById("cmTP");

    // Keyword Importances
    const kwModelName = document.getElementById("kwModelName");
    const spamKeywordsUl = document.getElementById("spamKeywordsUl");
    const hamKeywordsUl = document.getElementById("hamKeywordsUl");


    // ------------------------------------------------
    // 2. INPUT SYNC EVENTS (SLIDERS & CHAR COUNT)
    // ------------------------------------------------
    scanInput.addEventListener("input", () => {
        const len = scanInput.value.length;
        charCount.textContent = `${len} / 5000 characters`;
    });

    clearBtn.addEventListener("click", () => {
        scanInput.value = "";
        charCount.textContent = "0 / 5000 characters";
        scanInput.focus();
    });

    splitRatio.addEventListener("input", () => {
        splitRatioVal.textContent = `${splitRatio.value}%`;
    });

    nbAlpha.addEventListener("input", () => {
        nbAlphaVal.textContent = parseFloat(nbAlpha.value).toFixed(1);
    });

    svmC.addEventListener("input", () => {
        svmCVal.textContent = parseFloat(svmC.value).toFixed(1);
    });

    subsetSize.addEventListener("input", () => {
        const val = parseInt(subsetSize.value);
        if (val >= 5500) {
            subsetSizeVal.textContent = "Full Dataset (5574)";
        } else {
            subsetSizeVal.textContent = `${val} (Fast)`;
        }
    });


    // ------------------------------------------------
    // 3. TABS & SEGMENT CONTROL SWITCHING
    // ------------------------------------------------
    tabNB.addEventListener("click", () => {
        tabNB.classList.add("active");
        tabSVM.classList.remove("active");
        nbParamsSec.classList.remove("hidden");
        svmParamsSec.classList.add("hidden");
        activeModelTab = "naive_bayes";
    });

    tabSVM.addEventListener("click", () => {
        tabSVM.classList.add("active");
        tabNB.classList.remove("active");
        svmParamsSec.classList.remove("hidden");
        nbParamsSec.classList.add("hidden");
        activeModelTab = "svm";
    });

    // Predict Segmented Buttons
    btnPredNB.addEventListener("click", () => {
        btnPredNB.classList.add("active");
        btnPredSVM.classList.remove("active");
        activePredModel = "naive_bayes";
    });

    btnPredSVM.addEventListener("click", () => {
        btnPredSVM.classList.add("active");
        btnPredNB.classList.remove("active");
        activePredModel = "svm";
    });


    // ------------------------------------------------
    // 4. API DISPATCH: FETCH INITIAL MODEL STATUS
    // ------------------------------------------------
    async function updateStatus(isInitial = false) {
        try {
            const res = await fetch("/api/train/status");
            const data = await res.json();
            
            // Console update if active training is in progress
            renderConsoleLogs(data.console_logs);
            
            // Model active indicator update
            if (data.is_training) {
                statusMessageBadge.textContent = "Training...";
                statusMessageBadge.className = "badge badge-accent";
                trainBtn.disabled = true;
                trainBtnIcon.className = "fa-solid fa-spinner fa-spin";
                trainBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin" id="trainBtnIcon"></i> Training Pipeline...`;
                
                // Start polling if not already
                if (!isPollingStatus) {
                    startPolling();
                }
            } else {
                statusMessageBadge.textContent = data.status_message;
                statusMessageBadge.className = "badge badge-online";
                trainBtn.disabled = false;
                trainBtnIcon.className = "fa-solid fa-gear";
                trainBtn.innerHTML = `<i class="fa-solid fa-gear" id="trainBtnIcon"></i> Train Selected Model`;
                
                // Stop polling
                if (isPollingStatus) {
                    stopPolling();
                }
            }

            // Sync metrics comparison dashboard
            updateMetricsDashboard(data);

            // Sync Segment control prediction check
            const modelsTrained = data.models_trained;
            if (modelsTrained.naive_bayes) {
                btnPredNB.disabled = false;
                btnPredNB.title = "Naive Bayes model available.";
            } else {
                btnPredNB.disabled = true;
                btnPredNB.title = "Please train Naive Bayes first.";
            }

            if (modelsTrained.svm) {
                btnPredSVM.disabled = false;
                btnPredSVM.title = "SVM model available.";
            } else {
                btnPredSVM.disabled = true;
                btnPredSVM.title = "Please train SVM first.";
            }

            // Adjust active prediction selection if currently selected is disabled
            if (activePredModel === "naive_bayes" && !modelsTrained.naive_bayes && modelsTrained.svm) {
                btnPredSVM.click();
            } else if (activePredModel === "svm" && !modelsTrained.svm && modelsTrained.naive_bayes) {
                btnPredNB.click();
            }

        } catch (e) {
            console.error("Failed to fetch server status", e);
        }
    }

    function renderConsoleLogs(logs) {
        if (!logs || logs.length === 0) return;
        
        // Only re-render if count differs or changes
        const existingLinesCount = consoleLogs.children.length;
        if (existingLinesCount !== logs.length) {
            consoleLogs.innerHTML = "";
            logs.forEach(line => {
                const div = document.createElement("div");
                div.className = "terminal-line";
                div.textContent = line;
                consoleLogs.appendChild(div);
            });
            // Smooth scroll to bottom
            consoleLogs.scrollTop = consoleLogs.scrollHeight;
        }
    }

    function updateMetricsDashboard(data) {
        // Active model indicators
        const currentModel = data.current_model;
        if (currentModel === "naive_bayes") {
            statColNB.classList.add("active-model");
            statColSVM.classList.remove("active-model");
        } else if (currentModel === "svm") {
            statColSVM.classList.add("active-model");
            statColNB.classList.remove("active-model");
        } else {
            statColNB.classList.remove("active-model");
            statColSVM.classList.remove("active-model");
        }

        // 1. NAIVE BAYES METRICS
        const nbStats = data.metrics.naive_bayes;
        const nbTrained = data.models_trained.naive_bayes;
        
        if (nbTrained && nbStats) {
            trainedDotNB.className = "status-dot trained";
            accNB.textContent = (nbStats.accuracy * 100).toFixed(2) + "%";
            accBarNB.style.width = (nbStats.accuracy * 100).toFixed(0) + "%";
            
            precNB.textContent = (nbStats.precision * 100).toFixed(2) + "%";
            precBarNB.style.width = (nbStats.precision * 100).toFixed(0) + "%";
            
            recNB.textContent = (nbStats.recall * 100).toFixed(2) + "%";
            recBarNB.style.width = (nbStats.recall * 100).toFixed(0) + "%";
            
            f1NB.textContent = (nbStats.f1_score * 100).toFixed(2) + "%";
            f1BarNB.style.width = (nbStats.f1_score * 100).toFixed(0) + "%";
        } else {
            trainedDotNB.className = "status-dot";
            accNB.textContent = "--%";
            accBarNB.style.width = "0%";
            precNB.textContent = "--%";
            precBarNB.style.width = "0%";
            recNB.textContent = "--%";
            recBarNB.style.width = "0%";
            f1NB.textContent = "--%";
            f1BarNB.style.width = "0%";
        }

        // 2. SVM METRICS
        const svmStats = data.metrics.svm;
        const svmTrained = data.models_trained.svm;
        
        if (svmTrained && svmStats) {
            trainedDotSVM.className = "status-dot trained";
            accSVM.textContent = (svmStats.accuracy * 100).toFixed(2) + "%";
            accBarSVM.style.width = (svmStats.accuracy * 100).toFixed(0) + "%";
            
            precSVM.textContent = (svmStats.precision * 100).toFixed(2) + "%";
            precBarSVM.style.width = (svmStats.precision * 100).toFixed(0) + "%";
            
            recSVM.textContent = (svmStats.recall * 100).toFixed(2) + "%";
            recBarSVM.style.width = (svmStats.recall * 100).toFixed(0) + "%";
            
            f1SVM.textContent = (svmStats.f1_score * 100).toFixed(2) + "%";
            f1BarSVM.style.width = (svmStats.f1_score * 100).toFixed(0) + "%";
        } else {
            trainedDotSVM.className = "status-dot";
            accSVM.textContent = "--%";
            accBarSVM.style.width = "0%";
            precSVM.textContent = "--%";
            precBarSVM.style.width = "0%";
            recSVM.textContent = "--%";
            recBarSVM.style.width = "0%";
            f1SVM.textContent = "--%";
            f1BarSVM.style.width = "0%";
        }

        // 3. ADVANCED ANALYTICS (CONFUSION MATRIX & TOP WORDS OF ACTIVE MODEL)
        const activeModelName = currentModel === "naive_bayes" ? "Naive Bayes" : "SVM";
        cmModelName.textContent = activeModelName;
        kwModelName.textContent = activeModelName;

        const activePipeStats = currentModel === "naive_bayes" ? nbStats : svmStats;
        if (activePipeStats && activePipeStats.confusion_matrix) {
            const cm = activePipeStats.confusion_matrix;
            cmTN.textContent = cm.tn;
            cmFP.textContent = cm.fp;
            cmFN.textContent = cm.fn;
            cmTP.textContent = cm.tp;
        } else {
            cmTN.textContent = "0";
            cmFP.textContent = "0";
            cmFN.textContent = "0";
            cmTP.textContent = "0";
        }

        // Top keywords rendering
        renderKeywords(data.top_spam_words, spamKeywordsUl, true);
        renderKeywords(data.top_ham_words, hamKeywordsUl, false);
    }

    function renderKeywords(wordsList, targetUl, isSpam) {
        targetUl.innerHTML = "";
        
        if (!wordsList || wordsList.length === 0) {
            // Render nice mock empty lines
            for (let i = 0; i < 3; i++) {
                const li = document.createElement("li");
                li.className = "keyword-li";
                li.innerHTML = `<span class="kw-word">--</span> <span class="kw-val">--</span>`;
                targetUl.appendChild(li);
            }
            return;
        }

        // Limit to 5 top items for beautiful layout sizing
        const items = wordsList.slice(0, 5);
        items.forEach(item => {
            const li = document.createElement("li");
            li.className = "keyword-li";
            
            // Format score to 3 decimals
            const scoreStr = item.score >= 0 ? `+${item.score.toFixed(2)}` : `${item.score.toFixed(2)}`;
            li.innerHTML = `
                <span class="kw-word">${item.word}</span>
                <span class="kw-val" style="color: ${isSpam ? 'var(--spam-color)' : 'var(--ham-color)'}">${scoreStr}</span>
            `;
            targetUl.appendChild(li);
        });
    }

    // Polling helpers
    function startPolling() {
        isPollingStatus = true;
        pollIntervalId = setInterval(() => {
            updateStatus();
        }, 800);
    }

    function stopPolling() {
        isPollingStatus = false;
        if (pollIntervalId) {
            clearInterval(pollIntervalId);
            pollIntervalId = null;
        }
    }


    // ------------------------------------------------
    // 5. ASYNC DISPATCH: START MODEL TRAINING
    // ------------------------------------------------
    trainBtn.addEventListener("click", async () => {
        trainBtn.disabled = true;
        trainBtnIcon.className = "fa-solid fa-spinner fa-spin";
        
        // Assemble payloads
        const payload = {
            model_type: activeModelTab,
            max_features: parseInt(maxFeatures.value),
            stop_words: stopWords.value,
            ngram_range_max: parseInt(ngramRange.value),
            use_idf: true,
            split_ratio: parseFloat(splitRatio.value) / 100.0,
            nb_alpha: parseFloat(nbAlpha.value),
            svm_c: parseFloat(svmC.value),
            svm_kernel: svmKernel.value,
            subset_size: parseInt(subsetSize.value)
        };

        try {
            const res = await fetch("/api/train/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (data.success) {
                statusMessageBadge.textContent = "Training started...";
                statusMessageBadge.className = "badge badge-accent";
                // Start polling logs
                startPolling();
            } else {
                alert("Error starting training: " + data.message);
                trainBtn.disabled = false;
                trainBtnIcon.className = "fa-solid fa-gear";
            }
        } catch (e) {
            alert("Connection error: failed to start training pipeline.");
            trainBtn.disabled = false;
            trainBtnIcon.className = "fa-solid fa-gear";
        }
    });


    // ------------------------------------------------
    // 6. ASYNC DISPATCH: RUN INTERACTIVE SCAN (PREDICT)
    // ------------------------------------------------
    scanBtn.addEventListener("click", async () => {
        const text = scanInput.value.trim();
        if (!text) {
            alert("Please enter or select some email content to scan!");
            return;
        }

        scanBtn.disabled = true;
        scanBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Scanning Elements...`;

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    model: activePredModel
                })
            });
            
            const data = await res.json();
            
            if (data.success) {
                displayScanReport(data.prediction);
            } else {
                alert("Scan failed: " + data.error);
            }
        } catch (e) {
            alert("Scan failed due to server connection issue.");
        } finally {
            scanBtn.disabled = false;
            scanBtn.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles animate-sparkle"></i> Scan Email Content`;
        }
    });

    function displayScanReport(pred) {
        // Show Card
        resultsCard.classList.remove("hidden");
        // Smooth scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Update Text Info
        resultModelName.textContent = activePredModel === "naive_bayes" ? "Naive Bayes" : "SVM";
        resultConfidence.textContent = (pred.confidence * 100).toFixed(1) + "%";
        
        // Count words
        const words = pred.tokens.filter(t => t.is_word);
        resultWordCount.textContent = words.length;

        // Circular Progress Ring animation
        // Radius is 50, circumference is 314.159
        const percentVal = Math.round(pred.probability * 100);
        resultPercentage.textContent = `${percentVal}%`;
        
        const offset = 314.159 - (314.159 * pred.probability);
        resultRing.style.strokeDashoffset = offset;

        // Badge update
        resultBadge.textContent = pred.label;
        if (pred.label === "spam") {
            resultBadge.className = "classification-badge spam";
            resultRing.style.stroke = "var(--spam-color)";
        } else {
            resultBadge.className = "classification-badge ham";
            resultRing.style.stroke = "var(--ham-color)";
        }

        // 7. RENDER VISUAL WORD HIGHLIGHT CANVAS
        renderHighlightedCanvas(pred.tokens);
    }

    function renderHighlightedCanvas(tokens) {
        highlightedCanvas.innerHTML = "";
        
        // Normalize opacities so they are visually pleasing
        const scores = tokens.map(t => Math.abs(t.score));
        const maxScore = Math.max(...scores);
        
        tokens.forEach(tok => {
            if (!tok.is_word || tok.score === 0 || maxScore === 0) {
                // Common normal text token
                const textNode = document.createTextNode(tok.token);
                highlightedCanvas.appendChild(textNode);
            } else {
                // Highly active word token
                const span = document.createElement("span");
                
                // Opacity scaling (alpha bounds: 0.18 to 0.90)
                const relativeStrength = Math.abs(tok.score) / maxScore;
                const alpha = (relativeStrength * 0.72) + 0.18;
                
                const isSpammy = tok.score > 0;
                span.className = `hl-token ${isSpammy ? 'spammy' : 'hammy'}`;
                span.style.setProperty("--hl-alpha", alpha);
                
                // Floating point score layout
                const scoreSign = tok.score >= 0 ? `+${tok.score.toFixed(4)}` : `${tok.score.toFixed(4)}`;
                span.setAttribute("data-weight", `${tok.token} [weight: ${scoreSign}]`);
                span.textContent = tok.token;
                
                highlightedCanvas.appendChild(span);
            }
        });
    }

    closeResultsBtn.addEventListener("click", () => {
        resultsCard.classList.add("hidden");
    });


    // ------------------------------------------------
    // 7. PRESET LOADING & SELECTIONS
    // ------------------------------------------------
    async function loadPresets() {
        try {
            const res = await fetch("/api/samples");
            const samples = await res.json();
            samplesCached = samples;
            
            presetPills.innerHTML = "";
            samples.forEach((sample, index) => {
                const pill = document.createElement("span");
                pill.className = "preset-pill";
                pill.textContent = sample.title;
                pill.addEventListener("click", () => {
                    scanInput.value = sample.text;
                    // Trigger custom input change event for character count sync
                    const event = new Event('input', { bubbles: true });
                    scanInput.dispatchEvent(event);
                    
                    // Highlight pill briefly
                    document.querySelectorAll(".preset-pill").forEach(p => p.style.borderColor = "");
                    pill.style.borderColor = "var(--primary-color)";
                    
                    // Scroll to input
                    scanInput.focus();
                });
                presetPills.appendChild(pill);
            });
        } catch (e) {
            console.error("Failed to load preset sample templates", e);
            presetPills.innerHTML = `<span class="preset-pill" style="border-color: var(--spam-color)">Failed loading presets</span>`;
        }
    }


    // ------------------------------------------------
    // 8. ON-LOAD CONSTRUCTOR INITIALIZATION
    // ------------------------------------------------
    // Load static presets
    loadPresets();
    
    // Trigger initial status sync and start polling if already training
    updateStatus(true);
});
