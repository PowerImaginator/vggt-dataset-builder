import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "VGGT.DynamicInputs",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VGGT_Model_Inference") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Track the current number of image inputs (starts at 1 for image_1)
                this.imageInputCount = 1;
                
                // Method to add the next image input dynamically
                this.addNextImageInput = function() {
                    if (this.imageInputCount >= 20) return; // Max 20 images
                    
                    this.imageInputCount++;
                    const inputName = `image_${this.imageInputCount}`;
                    
                    // Check if input already exists
                    const existingInput = this.inputs?.find(i => i.name === inputName);
                    if (existingInput) return;
                    
                    // Add the new image input
                    this.addInput(inputName, "IMAGE");
                    
                    console.log(`[VGGT] Added ${inputName} input`);
                };
                
                // Method to remove unused image inputs
                this.cleanupImageInputs = function() {
                    // Find the highest connected image input
                    let highestConnected = 0;
                    
                    for (let i = 1; i <= this.imageInputCount; i++) {
                        const input = this.inputs?.find(inp => inp.name === `image_${i}`);
                        if (input && input.link != null) {
                            highestConnected = i;
                        }
                    }
                    
                    // Keep one extra input beyond the highest connected (minimum 1 for image_1)
                    const targetCount = Math.max(1, Math.min(highestConnected + 1, 20));
                    
                    // Remove inputs from the end down to targetCount
                    while (this.imageInputCount > targetCount) {
                        const inputName = `image_${this.imageInputCount}`;
                        const inputIndex = this.inputs?.findIndex(i => i.name === inputName);
                        
                        if (inputIndex !== undefined && inputIndex >= 0) {
                            this.removeInput(inputIndex);
                            console.log(`[VGGT] Removed ${inputName} input`);
                        }
                        
                        this.imageInputCount--;
                    }
                    
                    // Ensure the node updates visually
                    this.setSize?.(this.computeSize());
                };
                
                // Hook into connection changes to add/remove inputs dynamically
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function(type, index, connected, link_info) {
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    
                    // Only handle input connections
                    if (type === 1) { // type 1 = input
                        const input = this.inputs[index];
                        
                        if (input && input.name.startsWith("image_")) {
                            if (connected) {
                                // An image input was connected
                                const currentNum = parseInt(input.name.split("_")[1]);
                                
                                // If this is the highest image input, add the next one
                                if (currentNum === this.imageInputCount) {
                                    setTimeout(() => {
                                        this.addNextImageInput();
                                    }, 10);
                                }
                            } else {
                                // An image input was disconnected
                                setTimeout(() => {
                                    this.cleanupImageInputs();
                                }, 50);
                            }
                        }
                    }
                };
                
                // Also hook into onRemove to handle when links are deleted
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };
                
                return result;
            };
        }
    }
});
