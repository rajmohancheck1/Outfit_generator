import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import cv2
import os
import datetime
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

# Disable xformers to prevent import issues
os.environ["XFORMERS_DISABLED"] = "1"

# Try to import AI components with fallback
try:
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers imported successfully")
except ImportError as e:
    print(f"❌ Diffusers import failed: {e}")
    DIFFUSERS_AVAILABLE = False

try:
    from controlnet_aux import OpenposeDetector
    CONTROLNET_AUX_AVAILABLE = True
    print("✅ ControlNet aux imported successfully")
except ImportError as e:
    print(f"❌ ControlNet aux import failed: {e}")
    CONTROLNET_AUX_AVAILABLE = False

class OutfitGenerator:
    def __init__(self):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {self.device}")
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Create save directory
        self.save_folder = "saved_outfits"
        os.makedirs(self.save_folder, exist_ok=True)
        
        # Initialize AI models
        self.setup_models()
        
        # Style options
        self.genders = ["Female", "Male"]
        self.styles = ["Casual", "Formal", "Party", "Sports", "Traditional", "Modern"]
        self.seasons = ["Spring", "Summer", "Fall", "Winter", "All Season"]
        
    def setup_models(self):
        """Initialize the AI models for real outfit generation"""
        print("🔄 Setting up AI models...")
        
        if not DIFFUSERS_AVAILABLE:
            print("❌ Diffusers not available - running in demo mode")
            self.models_loaded = False
            return
            
        try:
            print("🤖 Loading AI models (this may take a few minutes)...")
            print("⚠️ XFormers disabled for compatibility")
            
            # Load ControlNet model for pose-guided generation
            print("📥 Loading ControlNet model...")
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-openpose",
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                low_cpu_mem_usage=True
            )
            
            # Load Stable Diffusion pipeline with ControlNet
            print("📥 Loading Stable Diffusion pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True
            )
            
            # Move to device and optimize
            print("🚀 Optimizing pipeline...")
            self.pipe = self.pipe.to(self.device)
            
            # Only enable optimizations if CUDA is available
            if self.device.type == 'cuda':
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("✅ Model CPU offload enabled")
                except Exception as e:
                    print(f"⚠️ Model CPU offload not available: {e}")
                    
                # Skip xformers completely to avoid compatibility issues
                print("⚠️ XFormers skipped for compatibility")
            else:
                print("🔄 Running on CPU - generation will be slower but stable")
            
            # Set scheduler for better quality
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            print("✅ Scheduler configured")
            
            # Initialize OpenPose detector if available
            if CONTROLNET_AUX_AVAILABLE:
                try:
                    print("📥 Loading OpenPose detector...")
                    self.openpose = OpenposeDetector.from_pretrained('lllyasviel/Annotators')
                    print("✅ OpenPose detector loaded")
                except Exception as e:
                    print(f"⚠️ OpenPose detector failed: {e}")
                    self.openpose = None
            else:
                self.openpose = None
            
            self.models_loaded = True
            print("🎉 AI models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            print("🎭 Falling back to demo mode...")
            self.models_loaded = False
    
    def extract_pose_controlnet(self, image):
        """Extract pose using OpenPose for ControlNet"""
        try:
            if not self.models_loaded or not hasattr(self, 'openpose') or self.openpose is None:
                return self.extract_pose_mediapipe(image)
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Extract pose using OpenPose
            pose_image = self.openpose(image)
            
            return pose_image, "Pose extracted successfully (OpenPose)"
            
        except Exception as e:
            print(f"OpenPose extraction failed: {e}")
            return self.extract_pose_mediapipe(image)
    
    def extract_pose_mediapipe(self, image):
        """Fallback pose extraction using MediaPipe"""
        try:
            # Convert PIL to numpy array
            if isinstance(image, Image.Image):
                image_rgb = np.array(image.convert("RGB"))
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(image_rgb)
            
            if not results.pose_landmarks:
                return None, "No pose detected. Please ensure full body is visible."
            
            # Create pose skeleton image
            pose_image = np.zeros_like(image_rgb)
            
            # Draw the pose landmarks
            self.mp_drawing.draw_landmarks(
                pose_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
            
            return Image.fromarray(pose_image), "Pose extracted successfully (MediaPipe)"
            
        except Exception as e:
            return None, f"Pose extraction failed: {str(e)}"
    
    def generate_outfit_prompt(self, gender, style, season, description):
        """Generate a comprehensive prompt for outfit generation"""
        # Base prompt for high quality fashion image
        base_prompt = f"professional fashion photography of a {gender.lower()} model"
        
        # Add specific clothing description
        if description.strip():
            base_prompt += f" wearing {description.strip()}"
        else:
            # Default outfit based on style
            style_defaults = {
                "Casual": "comfortable jeans and stylish top",
                "Formal": "elegant business attire",
                "Party": "trendy party outfit",
                "Sports": "athletic sportswear",
                "Traditional": "traditional cultural clothing",
                "Modern": "contemporary fashion outfit"
            }
            base_prompt += f" wearing {style_defaults.get(style, 'fashionable clothing')}"
        
        # Add style context
        base_prompt += f", {style.lower()} style"
        
        # Add seasonal context
        if season != "All Season":
            season_context = {
                "Spring": "light, fresh spring colors",
                "Summer": "bright, comfortable summer wear",
                "Fall": "warm autumn tones",
                "Winter": "cozy winter fashion"
            }
            base_prompt += f", {season_context.get(season, '')}"
        
        # Add quality and style modifiers
        base_prompt += ", high fashion, detailed clothing, perfect fit, studio lighting, 8k resolution, photorealistic"
        
        # Negative prompt for better quality
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, poorly drawn hands, poorly drawn face, mutation, mutilated, out of frame, bad art, beginner, amateur, distorted face"
        
        return base_prompt, negative_prompt
    
    def generate_outfit_ai(self, original_image, pose_image, prompt, negative_prompt):
        """Generate outfit using Stable Diffusion + ControlNet"""
        try:
            if not self.models_loaded:
                return self.simulate_outfit_generation(original_image, pose_image, prompt)
            
            # Ensure pose image is PIL
            if isinstance(pose_image, np.ndarray):
                pose_image = Image.fromarray(pose_image)
            
            print("🎨 Generating AI outfit... (this may take 30-60 seconds)")
            
            # Generate image using ControlNet
            generated_image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pose_image,
                num_inference_steps=15,  # Reduced for faster generation
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                width=512,
                height=768,
                generator=torch.manual_seed(42)
            ).images[0]
            
            print("✅ AI generation completed!")
            return generated_image, "AI outfit generated successfully!"
            
        except Exception as e:
            print(f"AI generation failed: {e}")
            return self.simulate_outfit_generation(original_image, pose_image, prompt)
    
    def simulate_outfit_generation(self, original_image, pose_image, prompt):
        """Fallback simulation for when AI models aren't available"""
        try:
            # Create a more sophisticated demo
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            
            # Create base for generated image
            width, height = original_image.size
            generated = Image.new('RGB', (width, height), (240, 240, 240))
            
            # Add pose outline
            if pose_image and isinstance(pose_image, Image.Image):
                # Blend pose with background
                pose_overlay = pose_image.convert('RGBA')
                generated.paste(pose_overlay, (0, 0), pose_overlay)
            
            # Add style-based color scheme
            draw = ImageDraw.Draw(generated)
            
            # Style-based modifications
            if "red" in prompt.lower() or "party" in prompt.lower():
                # Add red clothing simulation
                draw.rectangle([width//4, height//3, 3*width//4, 2*height//3], fill=(200, 50, 50))
            elif "blue" in prompt.lower() or "formal" in prompt.lower():
                # Add blue clothing simulation
                draw.rectangle([width//4, height//3, 3*width//4, 2*height//3], fill=(50, 50, 200))
            elif "sport" in prompt.lower():
                # Add athletic wear simulation
                draw.rectangle([width//4, height//3, 3*width//4, 2*height//3], fill=(100, 100, 100))
            
            # Add watermark
            try:
                draw.text((10, height-30), "Demo Mode - Install AI models for real generation", fill="orange")
            except:
                pass  # Skip if text drawing fails
            
            return generated, "Demo outfit generated (Install AI models for real generation)"
            
        except Exception as e:
            return None, f"Generation failed: {str(e)}"
    
    def process_image(self, image, gender, style, season, description):
        """Main processing function that combines all steps"""
        if image is None:
            return None, None, None, "Please upload or capture an image first."
        
        # Step 1: Extract pose
        pose_image, pose_status = self.extract_pose_controlnet(image)
        if pose_image is None:
            return None, None, None, pose_status
        
        # Step 2: Generate prompts
        prompt, negative_prompt = self.generate_outfit_prompt(gender, style, season, description)
        
        # Step 3: Generate outfit using AI
        outfit_image, gen_status = self.generate_outfit_ai(image, pose_image, prompt, negative_prompt)
        if outfit_image is None:
            return None, pose_image, prompt, gen_status
        
        return outfit_image, pose_image, prompt, f"✓ {pose_status}\n✓ {gen_status}"
    
    def save_outfit(self, image):
        """Save the generated outfit image"""
        if image is None:
            return "No image to save"
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outfit_{timestamp}.png"
        filepath = os.path.join(self.save_folder, filename)
        
        image.save(filepath)
        return f"✅ Saved as: {filepath}"

def create_ui():
    """Create the enhanced Gradio interface"""
    generator = OutfitGenerator()
    
    with gr.Blocks(title="AI Fashion Outfit Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 AI-Powered Custom Outfit Generator
        ### Real Stable Diffusion + ControlNet Fashion Technology
        Upload your photo to generate completely new outfits with AI!
        """)
        
        if not generator.models_loaded:
            gr.Markdown("""
            ⚠️ **Demo Mode**: AI models not fully loaded. To enable real AI generation:
            ```bash
            # First, completely remove xformers
            pip uninstall xformers -y
            
            # Then install required packages
            pip install diffusers transformers accelerate controlnet-aux
            ```
            """)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Input")
                
                input_image = gr.Image(
                    label="Upload Photo or Use Webcam",
                    type="pil",
                    sources=["upload", "webcam"],
                    height=400
                )
                
                with gr.Group():
                    gr.Markdown("### 👤 Style Preferences")
                    gender = gr.Dropdown(
                        choices=generator.genders,
                        label="Gender", 
                        value="Female"
                    )
                    style = gr.Dropdown(
                        choices=generator.styles,
                        label="Style",
                        value="Casual"
                    )
                    season = gr.Dropdown(
                        choices=generator.seasons,
                        label="Season",
                        value="All Season"
                    )
                    description = gr.Textbox(
                        label="Detailed Outfit Description",
                        placeholder="e.g., elegant red evening dress with silver jewelry, casual blue jeans with white t-shirt, black business suit...",
                        lines=3
                    )
                
                generate_btn = gr.Button("🎨 Generate New Outfit", variant="primary", size="lg")
                save_btn = gr.Button("💾 Save Generated Outfit", variant="secondary")
            
            # Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 AI Generated Results")
                
                output_image = gr.Image(
                    label="🎨 AI Generated Outfit",
                    type="pil",
                    height=400,
                    show_download_button=True
                )
                
                pose_image = gr.Image(
                    label="🤖 Detected Pose Structure",
                    type="pil", 
                    height=250
                )
                
                status_text = gr.Textbox(
                    label="🔄 Generation Status",
                    lines=3,
                    interactive=False
                )
                
                generated_prompt = gr.Textbox(
                    label="📝 AI Prompt Used",
                    lines=3,
                    interactive=False
                )
                
                save_status = gr.Textbox(
                    label="💾 Save Status",
                    interactive=False
                )
        
        # Instructions
        with gr.Row():
            gr.Markdown("""
            ### 💡 Tips for Best Results:
            - **High Quality Photos**: Use clear, well-lit, full-body photos
            - **Neutral Pose**: Stand straight facing camera for best pose detection  
            - **Detailed Descriptions**: Be specific about colors, styles, and clothing items
            - **Experiment**: Try different combinations and descriptions for variety
            - **Current Status**: The app is running in compatibility mode (XFormers disabled for stability)
            """)
        
        # Event handlers
        generate_btn.click(
            fn=generator.process_image,
            inputs=[input_image, gender, style, season, description],
            outputs=[output_image, pose_image, generated_prompt, status_text],
            show_progress=True
        )
        
        save_btn.click(
            fn=generator.save_outfit,
            inputs=[output_image],
            outputs=[save_status]
        )
        
        # Enhanced examples
        gr.Examples(
            examples=[
                ["Female", "Party", "Summer", "elegant red cocktail dress with high heels and gold jewelry"],
                ["Male", "Formal", "All Season", "navy blue business suit with white shirt and silk tie"],
                ["Female", "Casual", "Spring", "light blue jeans with pastel pink sweater and white sneakers"],
                ["Male", "Sports", "Summer", "black athletic shorts with neon green running shirt and training shoes"],
                ["Female", "Traditional", "Fall", "beautiful floral saree with traditional jewelry"],
                ["Male", "Modern", "Winter", "stylish black leather jacket with dark jeans and boots"]
            ],
            inputs=[gender, style, season, description],
            label="Try These Example Prompts"
        )
    
    return demo

# Installation check and startup
def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking AI components...")
    
    if DIFFUSERS_AVAILABLE:
        print("✅ Diffusers available")
    else:
        print("❌ Diffusers not available")
        
    if CONTROLNET_AUX_AVAILABLE:
        print("✅ ControlNet auxiliary tools available")
    else:
        print("❌ ControlNet auxiliary tools not available")
    
    # Check PyTorch configuration
    print(f"🔧 PyTorch device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️ Running on CPU - generation will be slower")
    
    print(f"🛡️ XFormers disabled for compatibility")
    
    if not DIFFUSERS_AVAILABLE:
        print("\n🔧 To enable full AI functionality:")
        print("1. Completely remove xformers: pip uninstall xformers -y")
        print("2. Install required packages: pip install diffusers transformers accelerate controlnet-aux")
        print("\n🎨 Running in demo mode for now...")

if __name__ == "__main__":
    print("🚀 Starting AI Outfit Generator...")
    print("🛡️ XFormers compatibility mode enabled")
    check_requirements()
    
    demo = create_ui()
    
    if DIFFUSERS_AVAILABLE:
        print("🎨 Upload a photo and describe your desired outfit!")
        print("🤖 AI will generate completely new clothing based on your description")
    else:
        print("🎭 Running in demo mode - fix dependencies for full AI generation")
    
    demo.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        inbrowser=True
    )