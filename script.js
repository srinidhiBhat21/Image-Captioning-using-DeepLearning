class BLIPImageCaptioning {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.loadHistory();
        this.checkBackendHealth();
    }

    initializeElements() {
        // Get all DOM elements
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.promptInput = document.getElementById('promptInput');
        this.generateBtn = document.getElementById('generateBtn');
        this.resultSection = document.getElementById('resultSection');
        this.previewImage = document.getElementById('previewImage');
        this.captionText = document.getElementById('captionText');
        this.inferenceTime = document.getElementById('inferenceTime');
        this.deviceInfo = document.getElementById('deviceInfo');
        this.loadingSpinner = document.getElementById('loadingSpinner');
        this.performanceInfo = document.getElementById('performanceInfo');
        this.copyBtn = document.getElementById('copyBtn');
        this.shareBtn = document.getElementById('shareBtn');
        this.historyList = document.getElementById('historyList');
        this.clearHistoryBtn = document.getElementById('clearHistoryBtn');
        this.imageSize = document.getElementById('imageSize');
        this.imageDimensions = document.getElementById('imageDimensions');
        this.toast = document.getElementById('toast');
        this.toastMessage = document.getElementById('toastMessage');
        
        // Initialize data
        this.selectedFile = null;
        this.history = [];
        this.backendUrl = 'http://localhost:5000';
    }

    setupEventListeners() {
        // Upload area events
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        // File input change
        this.imageInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Generate button
        this.generateBtn.addEventListener('click', this.generateCaption.bind(this));
        
        // Action buttons
        this.copyBtn.addEventListener('click', this.copyCaption.bind(this));
        this.shareBtn.addEventListener('click', this.shareCaption.bind(this));
        this.clearHistoryBtn.addEventListener('click', this.clearHistory.bind(this));
        
        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboard.bind(this));
    }

    handleKeyboard(e) {
        // Ctrl+V to paste image from clipboard
        if (e.ctrlKey && e.key === 'v') {
            this.handleClipboardPaste(e);
        }
        
        // Enter to generate caption if image is selected
        if (e.key === 'Enter' && this.selectedFile) {
            this.generateCaption();
        }
    }

    async handleClipboardPaste(e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                const blob = item.getAsFile();
                this.processFile(blob);
                break;
            }
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showToast('Please select a valid image file.', 'error');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showToast('File size must be less than 10MB.', 'error');
            return;
        }

        this.selectedFile = file;
        this.generateBtn.disabled = false;

        // Show file info
        this.imageSize.textContent = `Size: ${(file.size / 1024 / 1024).toFixed(2)} MB`;

        // Preview the image
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.previewImage.onload = () => {
                this.imageDimensions.textContent = 
                    `Dimensions: ${this.previewImage.naturalWidth} × ${this.previewImage.naturalHeight}`;
            };
            
            this.resultSection.style.display = 'block';
            this.captionText.textContent = 'Click "Generate Caption" to analyze this image with BLIP AI model.';
            this.performanceInfo.style.display = 'none';
            
            // Smooth scroll to results
            this.resultSection.scrollIntoView({ behavior: 'smooth' });
        };
        reader.readAsDataURL(file);

        this.showToast('Image uploaded successfully!', 'success');
    }

    async generateCaption() {
        if (!this.selectedFile) return;

        // Show loading state
        this.generateBtn.disabled = true;
        this.loadingSpinner.style.display = 'flex';
        this.captionText.textContent = 'AI is analyzing your image using the BLIP deep learning model...';
        this.performanceInfo.style.display = 'none';

        try {
            // Prepare form data
            const formData = new FormData();
            formData.append('image', this.selectedFile);
            
            const prompt = this.promptInput.value.trim();
            if (prompt) {
                formData.append('prompt', prompt);
            }

            const startTime = Date.now();

            // Make API call to backend
            const response = await fetch(`${this.backendUrl}/api/caption`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (!data.success) {
                throw new Error(data.error || 'Unknown error occurred');
            }

            const endTime = Date.now();
            const totalTime = (endTime - startTime) / 1000;

            // Display results
            this.captionText.textContent = data.caption;
            this.inferenceTime.textContent = `${data.inference_time}s`;
            this.performanceInfo.style.display = 'flex';

            // Add to history
            this.addToHistory({
                image: this.previewImage.src,
                caption: data.caption,
                timestamp: new Date().toLocaleString(),
                inferenceTime: data.inference_time,
                prompt: prompt || 'No prompt',
                fileName: this.selectedFile.name
            });

            this.showToast('Caption generated successfully!', 'success');

        } catch (error) {
            console.error('Error generating caption:', error);
            this.captionText.textContent = `Error: ${error.message}. Please check if the backend server is running.`;
            this.showToast('Error generating caption. Check console for details.', 'error');
        } finally {
            // Hide loading state
            this.loadingSpinner.style.display = 'none';
            this.generateBtn.disabled = false;
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.backendUrl}/health`);
            if (response.ok) {
                const data = await response.json();
                this.deviceInfo.textContent = data.device || 'Unknown';
                console.log(' Backend connected successfully');
            }
        } catch (error) {
            console.warn(' Backend not available. Please start the Flask server.');
            this.deviceInfo.textContent = 'Backend offline';
        }
    }

    copyCaption() {
        const caption = this.captionText.textContent;
        if (caption && caption !== 'Click "Generate Caption" to analyze this image with BLIP AI model.') {
            navigator.clipboard.writeText(caption).then(() => {
                this.showToast('Caption copied to clipboard!', 'success');
                
                // Visual feedback on button
                const originalHTML = this.copyBtn.innerHTML;
                this.copyBtn.innerHTML = '<i class="fas fa-check"></i> <span>Copied!</span>';
                setTimeout(() => {
                    this.copyBtn.innerHTML = originalHTML;
                }, 2000);
            });
        }
    }

    shareCaption() {
        const caption = this.captionText.textContent;
        if (caption && caption !== 'Click "Generate Caption" to analyze this image with BLIP AI model.') {
            if (navigator.share) {
                navigator.share({
                    title: 'AI Generated Image Caption',
                    text: caption
                });
            } else {
                // Fallback: copy to clipboard
                this.copyCaption();
            }
        }
    }

    addToHistory(item) {
        this.history.unshift(item);
        if (this.history.length > 10) {
            this.history = this.history.slice(0, 10);
        }
        this.saveHistory();
        this.renderHistory();
    }

    saveHistory() {
        localStorage.setItem('blipCaptionHistory', JSON.stringify(this.history));
    }

    loadHistory() {
        const saved = localStorage.getItem('blipCaptionHistory');
        if (saved) {
            this.history = JSON.parse(saved);
            this.renderHistory();
        }
    }

    renderHistory() {
        if (this.history.length === 0) {
            this.historyList.innerHTML = `
                <div class="empty-history">
                    <i class="fas fa-image"></i>
                    <p>No captions generated yet. Upload an image to get started!</p>
                </div>
            `;
            return;
        }

        this.historyList.innerHTML = '';
        
        this.history.forEach((item, index) => {
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            historyItem.innerHTML = `
                <img src="${item.image}" alt="Historical image" class="history-thumbnail">
                <div class="history-content">
                    <div class="history-caption">"${item.caption}"</div>
                    <div class="history-meta">
                        <span><i class="fas fa-clock"></i> ${item.timestamp}</span>
                        <span><i class="fas fa-zap"></i> ${item.inferenceTime}s</span>
                        <span><i class="fas fa-file-image"></i> ${item.fileName}</span>
                    </div>
                    ${item.prompt !== 'No prompt' ? `<div style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;"><i class="fas fa-magic"></i> Prompt: "${item.prompt}"</div>` : ''}
                </div>
            `;
            
            // Add click event to copy caption
            historyItem.addEventListener('click', () => {
                navigator.clipboard.writeText(item.caption);
                this.showToast('Caption copied from history!', 'success');
            });
            
            this.historyList.appendChild(historyItem);
        });
    }

    clearHistory() {
        if (confirm('Are you sure you want to clear all caption history?')) {
            this.history = [];
            this.saveHistory();
            this.renderHistory();
            this.showToast('History cleared successfully!', 'success');
        }
    }

    showToast(message, type = 'success') {
        this.toastMessage.textContent = message;
        this.toast.className = `toast ${type}`;
        this.toast.classList.add('show');
        
        setTimeout(() => {
            this.toast.classList.remove('show');
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new BLIPImageCaptioning();
});
