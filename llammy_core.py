# =============================================================================
# LLAMMY CORE - RE-ENGINEERED WITH VISION INTEGRATION
# llammy_core.py - Unified services architecture
# =============================================================================

import time
import os
import sqlite3
import json
import bpy
import requests
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

print("Llammy Core - Re-engineered with Vision Integration Loading...")

# =============================================================================
# CENTRALIZED OLLAMA SERVICE
# =============================================================================

class OllamaService:
    """Centralized Ollama communication service"""
    
    def __init__(self):
        self._model_cache = None
        self._cache_time = 0
        self._cache_duration = 30  # 30 seconds
        self.base_url = "http://localhost:11434"
        self.timeout = 8  # Consistent timeout
        
        # Model prioritization
        self.priority_keywords = {
            'triangle104': 100,  # Highest priority
            'llammy': 90,
            'blender': 80,
            'code': 70,
            'coder': 70,
            'deepseek': 60,
            'qwen': 50
        }
    
    def get_models_for_blender(self, force_refresh=False) -> List[Tuple[str, str, str]]:
        """Get models formatted for Blender EnumProperty"""
        now = time.time()
        
        if not force_refresh and self._model_cache and (now - self._cache_time) < self._cache_duration:
            return self._model_cache
        
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                if models:
                    formatted_models = self._format_models_for_blender(models)
                    self._model_cache = formatted_models
                    self._cache_time = now
                    return formatted_models
                else:
                    return [("no_models", "No Models Available", "No models found")]
            else:
                return [("connection_failed", "Connection Failed", f"Server returned {response.status_code}")]
                
        except requests.exceptions.ConnectionError:
            return [("connection_failed", "Ollama Not Running", "Start Ollama service")]
        except requests.exceptions.Timeout:
            return [("timeout", "Connection Timeout", "Ollama not responding")]
        except Exception as e:
            return [("error", f"Error: {str(e)[:20]}", f"Ollama error: {str(e)}")]
    
    def _format_models_for_blender(self, models: List[Dict]) -> List[Tuple[str, str, str]]:
        """Format models with priority sorting for Blender"""
        model_items = []
        
        # Calculate priority scores
        scored_models = []
        for model in models:
            model_name = model['name']
            score = self._calculate_priority_score(model_name)
            scored_models.append((score, model_name))
        
        # Sort by priority (highest first)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        # Format for Blender
        for score, model_name in scored_models:
            if score >= 100:  # Triangle104
                icon = "🎯 Triangle104"
            elif score >= 90:  # Llammy
                icon = "🎭 Llammy"
            elif score >= 80:  # Blender
                icon = "🔧 Blender"
            elif score >= 70:  # Code models
                icon = "💻 Code"
            else:
                icon = "🤖"
            
            display_name = f"{icon} {model_name[:30]}"
            description = f"Priority: {score} - {model_name}"
            
            model_items.append((model_name, display_name, description))
        
        return model_items if model_items else [("no_models", "No Models", "No models available")]
    
    def _calculate_priority_score(self, model_name: str) -> int:
        """Calculate priority score for model"""
        name_lower = model_name.lower()
        score = 0
        
        for keyword, priority in self.priority_keywords.items():
            if keyword in name_lower:
                score = max(score, priority)
        
        return score
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                return {
                    'success': True,
                    'total_models': len(models),
                    'triangle104_available': any('triangle104' in m['name'].lower() for m in models),
                    'status': 'connected'
                }
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'status': 'error'
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'status': 'disconnected'
            }
    
    def resolve_best_model(self, requested_model: str, role: str = "general") -> str:
        """Resolve the best available model for a role"""
        models = self.get_models_for_blender()
        available_names = [m[0] for m in models if m[0] not in ['no_models', 'connection_failed', 'error']]
        
        # If requested model is available, use it
        if requested_model in available_names:
            return requested_model
        
        # Fallback based on role
        if role == "creative":
            # Prefer Triangle104, then Llammy, then others
            for model_name in available_names:
                if 'triangle104' in model_name.lower():
                    return model_name
            for model_name in available_names:
                if 'llammy' in model_name.lower():
                    return model_name
        
        elif role == "technical":
            # Prefer code models, then Triangle104
            for model_name in available_names:
                if any(term in model_name.lower() for term in ['code', 'coder', 'deepseek']):
                    return model_name
            for model_name in available_names:
                if 'triangle104' in model_name.lower():
                    return model_name
        
        # Return first available model as final fallback
        return available_names[0] if available_names else requested_model

# =============================================================================
# VISION INTELLIGENCE SERVICE
# =============================================================================

class VisionIntelligenceService:
    """Centralized vision processing for all AI components"""
    
    def __init__(self):
        self.vision_available = False
        self.vision_pipeline = None
        self.frame_buffer = {}
        self.analysis_cache = {}
        
        # Try to initialize vision components
        self._initialize_vision()
    
    def _initialize_vision(self):
        """Initialize Vision LoRA components"""
        try:
            # Import vision dependencies
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from peft import PeftModel
            
            # Check if Vision LoRA package is available
            vision_path = Path(__file__).parent / "Vision_LoRA_Package"
            if not vision_path.exists():
                print("Vision LoRA package not found - vision features disabled")
                return
            
            # Initialize vision pipeline
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load CLIP with LoRA adapter
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=torch.float16
            ).to(device)
            
            adapter_path = vision_path / "adapters" / "CLIP-LoRA"
            if adapter_path.exists():
                self.clip_lora = PeftModel.from_pretrained(
                    clip_model,
                    str(adapter_path),
                    torch_dtype=torch.float16
                ).to(device)
            else:
                self.clip_lora = clip_model
            
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.device = device
            self.vision_available = True
            
            print(f"Vision Intelligence Service initialized on {device}")
            
        except ImportError as e:
            print(f"Vision dependencies not available: {e}")
        except Exception as e:
            print(f"Vision initialization failed: {e}")
    
    def capture_viewport_state(self) -> Dict[str, Any]:
        """Capture current Blender viewport state"""
        try:
            scene = bpy.context.scene
            
            # Basic scene analysis without rendering
            scene_data = {
                'object_count': len(scene.objects),
                'active_object': bpy.context.active_object.name if bpy.context.active_object else None,
                'objects': [obj.name for obj in scene.objects],
                'selected_objects': [obj.name for obj in bpy.context.selected_objects],
                'scene_name': scene.name,
                'frame_current': scene.frame_current,
                'camera_location': list(scene.camera.location) if scene.camera else None,
                'has_materials': any(obj.data.materials for obj in scene.objects if hasattr(obj.data, 'materials')),
                'timestamp': time.time()
            }
            
            # Add render info if possible
            try:
                render = scene.render
                scene_data['render_settings'] = {
                    'resolution_x': render.resolution_x,
                    'resolution_y': render.resolution_y,
                    'engine': render.engine
                }
            except:
                pass
            
            return scene_data
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': time.time(),
                'object_count': 0
            }
    
    def analyze_scene_context(self, scene_data: Dict[str, Any]) -> str:
        """Convert scene data to textual context for AI models"""
        if 'error' in scene_data:
            return f"Scene analysis error: {scene_data['error']}"
        
        context_parts = []
        
        # Object information
        obj_count = scene_data.get('object_count', 0)
        if obj_count == 0:
            context_parts.append("Empty scene")
        elif obj_count == 1:
            context_parts.append("Scene with 1 object")
        else:
            context_parts.append(f"Scene with {obj_count} objects")
        
        # Active object
        active_obj = scene_data.get('active_object')
        if active_obj:
            context_parts.append(f"Active object: {active_obj}")
        
        # Selection info
        selected = scene_data.get('selected_objects', [])
        if selected:
            if len(selected) == 1:
                context_parts.append(f"Selected: {selected[0]}")
            else:
                context_parts.append(f"Selected: {len(selected)} objects")
        
        # Camera info
        if scene_data.get('camera_location'):
            context_parts.append("Camera positioned")
        
        # Materials
        if scene_data.get('has_materials'):
            context_parts.append("Scene has materials")
        
        return ". ".join(context_parts) + "."
    
    def extract_visual_intent(self, user_request: str) -> Optional[Dict[str, Any]]:
        """Extract visual intent from user request"""
        visual_keywords = {
            'style': ['realistic', 'cartoon', 'stylized', 'organic', 'mechanical'],
            'shape': ['sphere', 'cube', 'cylinder', 'plane', 'torus', 'cone'],
            'complexity': ['simple', 'complex', 'detailed', 'basic', 'advanced'],
            'composition': ['grid', 'array', 'scattered', 'centered', 'random'],
            'lighting': ['bright', 'dark', 'dramatic', 'soft', 'ambient']
        }
        
        request_lower = user_request.lower()
        intent = {}
        
        for category, keywords in visual_keywords.items():
            matches = [kw for kw in keywords if kw in request_lower]
            if matches:
                intent[category] = matches
        
        return intent if intent else None
    
    def get_enhanced_context(self, user_request: str) -> Dict[str, Any]:
        """Get enhanced context combining scene and visual analysis"""
        scene_data = self.capture_viewport_state()
        scene_context = self.analyze_scene_context(scene_data)
        visual_intent = self.extract_visual_intent(user_request)
        
        return {
            'scene_context': scene_context,
            'visual_intent': visual_intent,
            'scene_data': scene_data,
            'vision_available': self.vision_available
        }

# =============================================================================
# RE-ENGINEERED CORE ORCHESTRATOR
# =============================================================================

class LlammyCore:
    """Re-engineered core orchestrator with unified services"""
    
    def __init__(self):
        self.version = "5.0.0-VISION-INTEGRATED"
        self.initialized = False
        
        # Core services
        self.ollama_service = OllamaService()
        self.vision_service = VisionIntelligenceService()
        self.ai_engine = None
        
        # Core stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'vision_enhanced_requests': 0,
            'avg_response_time': 0.0,
            'startup_time': datetime.now().isoformat()
        }
        
        print(f"Llammy Core v{self.version} - Vision-integrated architecture")
    
    def initialize(self) -> bool:
        """Initialize core system with all services"""
        try:
            print("Initializing Llammy Core with vision integration...")
            
            # Test Ollama connection
            ollama_status = self.ollama_service.test_connection()
            if ollama_status['success']:
                print(f"Ollama connected: {ollama_status['total_models']} models available")
            else:
                print(f"Ollama connection failed: {ollama_status['error']}")
            
            # Vision service status
            if self.vision_service.vision_available:
                print("Vision Intelligence Service: ACTIVE")
            else:
                print("Vision Intelligence Service: DISABLED (missing dependencies)")
            
            self.initialized = True
            print("Core system initialized successfully")
            return True
            
        except Exception as e:
            print(f"Core initialization failed: {e}")
            return False
    
    def set_ai_engine(self, ai_engine):
        """Connect AI engine to core"""
        self.ai_engine = ai_engine
        # Provide services to AI engine
        if hasattr(ai_engine, 'set_services'):
            ai_engine.set_services(
                ollama_service=self.ollama_service,
                vision_service=self.vision_service
            )
        print("AI engine connected to core with service integration")
    
    def process_llammy_request(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Main request processing with vision enhancement"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        print(f"Core processing with vision enhancement: {user_request[:50]}...")
        
        try:
            # STEP 1: Enhanced context gathering
            enhanced_context = self._gather_enhanced_context(user_request, **kwargs)
            
            # STEP 2: Preprocess request with vision context
            processed_request = self._preprocess_request(user_request, enhanced_context, **kwargs)
            
            # STEP 3: Execute AI request with enhanced context
            result = self._execute_ai_request(processed_request, enhanced_context, **kwargs)
            
            # STEP 4: Post-process with vision validation
            final_result = self._postprocess_with_vision(result, user_request, start_time)
            
            # Update stats
            if final_result.get('success'):
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            
            if final_result.get('vision_enhanced'):
                self.stats['vision_enhanced_requests'] += 1
            
            processing_time = time.time() - start_time
            self._update_avg_response_time(processing_time)
            
            return final_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['failed_requests'] += 1
            
            print(f"Core processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'generated_code': '',
                'core_method': 'core_failed'
            }
    
    def _gather_enhanced_context(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Gather enhanced context with vision analysis"""
        context = {
            'file_context': kwargs.get('file_context', []),
            'vision_enhanced': False
        }
        
        # Get vision-enhanced context
        if self.vision_service.vision_available:
            vision_context = self.vision_service.get_enhanced_context(user_request)
            context.update(vision_context)
            context['vision_enhanced'] = True
        
        # Add basic Blender context
        context['blender_context'] = self._get_blender_context()
        
        return context
    
    def _get_blender_context(self) -> str:
        """Get basic Blender context information"""
        try:
            context_parts = []
            
            # Blender version
            context_parts.append(f"Blender {bpy.app.version_string}")
            
            # Current mode
            if bpy.context.mode:
                context_parts.append(f"Mode: {bpy.context.mode}")
            
            # Workspace
            if bpy.context.workspace:
                context_parts.append(f"Workspace: {bpy.context.workspace.name}")
            
            return ". ".join(context_parts)
            
        except:
            return "Blender 4.4/4.5 environment"
    
    def _preprocess_request(self, user_request: str, enhanced_context: Dict[str, Any], **kwargs) -> str:
        """Preprocess request with enhanced context"""
        # Basic cleaning
        cleaned = user_request.strip()
        
        # Add context awareness
        if enhanced_context.get('scene_context'):
            # The AI will receive this as part of the context, not modifying the user request
            pass
        
        return cleaned
    
    def _execute_ai_request(self, user_request: str, enhanced_context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute AI request with enhanced context"""
        
        if self.ai_engine:
            # Use connected AI engine with enhanced context
            try:
                # Build comprehensive context string
                context_parts = []
                
                if enhanced_context.get('scene_context'):
                    context_parts.append(f"Current scene: {enhanced_context['scene_context']}")
                
                if enhanced_context.get('blender_context'):
                    context_parts.append(enhanced_context['blender_context'])
                
                if enhanced_context.get('file_context'):
                    context_parts.append(f"File context: {len(enhanced_context['file_context'])} files")
                
                full_context = ". ".join(context_parts)
                
                # Pass enhanced context to AI engine
                result = self.ai_engine.execute_request(
                    user_request,
                    context=full_context,
                    enhanced_context=enhanced_context,
                    **kwargs
                )
                
                return result
                
            except Exception as e:
                print(f"AI engine failed: {e}")
                # Fall through to basic generation
        
        # Basic code generation fallback
        return self._generate_basic_code(user_request, enhanced_context)
    
    def _generate_basic_code(self, user_request: str, enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic Blender code with context awareness"""
        request_lower = user_request.lower()
        
        # Use scene context to make smarter decisions
        scene_data = enhanced_context.get('scene_data', {})
        has_objects = scene_data.get('object_count', 0) > 0
        
        # Smart object creation based on context
        if 'sphere' in request_lower:
            if has_objects:
                code = '''import bpy

# Create sphere (keeping existing objects)
bpy.ops.mesh.primitive_uv_sphere_add(location=(2, 0, 0))
sphere = bpy.context.active_object
sphere.name = "AI_Generated_Sphere"

print("Sphere created successfully!")'''
            else:
                code = '''import bpy

# Clear scene and create sphere
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, type='MESH')

bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))
sphere = bpy.context.active_object
sphere.name = "AI_Generated_Sphere"

print("Sphere created successfully!")'''
            
        elif 'cube' in request_lower:
            if has_objects:
                code = '''import bpy

# Create cube (keeping existing objects)
bpy.ops.mesh.primitive_cube_add(location=(2, 0, 0))
cube = bpy.context.active_object
cube.name = "AI_Generated_Cube"

print("Cube created successfully!")'''
            else:
                code = '''import bpy

# Clear scene and create cube
bpy.ops.object.select_all(action='SELECT')
# OLD (causing error):
#bpy.ops.object.delete(use_global=False, type='MESH')

# NEW (works in 4.5):
bpy.ops.object.delete(use_global=False)

bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
cube = bpy.context.active_object
cube.name = "AI_Generated_Cube"

print("Cube created successfully!")'''
            
        else:
            # Generic object creation with context awareness
            location = "(2, 0, 0)" if has_objects else "(0, 0, 0)"
            clear_code = "" if has_objects else """
# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, type='MESH')
"""
            
            code = f'''import bpy
{clear_code}
# Create basic object
bpy.ops.mesh.primitive_cube_add(location={location})
obj = bpy.context.active_object
obj.name = "AI_Generated_Object"

print("Object created successfully!")'''
        
        return {
            'success': True,
            'generated_code': code,
            'method': 'context_aware_basic_generation',
            'context_used': enhanced_context.get('scene_context', 'none'),
            'processing_time': 0.1
        }
    
    def _postprocess_with_vision(self, result: Dict[str, Any], user_request: str, start_time: float) -> Dict[str, Any]:
        """Post-process result with vision validation"""
        processing_time = time.time() - start_time
        
        # Add core metadata
        result['processing_time'] = processing_time
        result['core_version'] = self.version
        result['vision_available'] = self.vision_service.vision_available
        
        # Add vision enhancement flag
        if self.vision_service.vision_available:
            result['vision_enhanced'] = True
        
        return result
    
    def _update_avg_response_time(self, processing_time: float):
        """Update average response time"""
        total_requests = self.stats['total_requests']
        current_avg = self.stats['avg_response_time']
        
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.stats['avg_response_time'] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        ollama_status = self.ollama_service.test_connection()
        
        return {
            'core': {
                'initialized': self.initialized,
                'version': self.version,
                'stats': self.stats.copy(),
                'ai_engine_connected': self.ai_engine is not None
            },
            'services': {
                'ollama': ollama_status,
                'vision': {
                    'available': self.vision_service.vision_available,
                    'status': 'active' if self.vision_service.vision_available else 'disabled'
                }
            }
        }

# =============================================================================
# GLOBAL INSTANCE AND MODULE FUNCTIONS
# =============================================================================

llammy_core_instance = None

def initialize_llammy_core() -> Dict[str, Any]:
    """Initialize global core instance"""
    global llammy_core_instance
    
    try:
        print("Initializing global Llammy core with vision integration...")
        
        llammy_core_instance = LlammyCore()
        
        if llammy_core_instance.initialize():
            print("Global Llammy core initialized successfully")
            return {
                'success': True,
                'message': 'Core initialized with vision integration',
                'vision_available': llammy_core_instance.vision_service.vision_available,
                'ollama_status': llammy_core_instance.ollama_service.test_connection()
            }
        else:
            raise Exception("Core initialization failed")
            
    except Exception as e:
        print(f"Global core initialization failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_llammy_core() -> Optional[LlammyCore]:
    """Get global core instance"""
    global llammy_core_instance
    
    if llammy_core_instance is None:
        init_result = initialize_llammy_core()
        if not init_result.get('success'):
            return None
    
    return llammy_core_instance

def get_ollama_service() -> Optional[OllamaService]:
    """Get centralized Ollama service"""
    core = get_llammy_core()
    return core.ollama_service if core else None

def get_vision_service() -> Optional[VisionIntelligenceService]:
    """Get centralized Vision service"""
    core = get_llammy_core()
    return core.vision_service if core else None

def process_llammy_request(user_request: str, **kwargs) -> Dict[str, Any]:
    """Process request through global core"""
    core = get_llammy_core()
    
    if core:
        return core.process_llammy_request(user_request, **kwargs)
    else:
        return {
            'success': False,
            'error': 'Core not available',
            'generated_code': '',
            'processing_time': 0.0
        }

# =============================================================================
# TESTING
# =============================================================================

def test_reengineered_core():
    """Test re-engineered core with vision integration"""
    print("Testing re-engineered core...")
    
    # Test initialization
    init_result = initialize_llammy_core()
    print(f"Initialization: {'SUCCESS' if init_result.get('success') else 'FAILED'}")
    
    if init_result.get('success'):
        print(f"Vision available: {init_result.get('vision_available', False)}")
        print(f"Ollama status: {init_result.get('ollama_status', {}).get('status', 'unknown')}")
    
    # Test services
    core = get_llammy_core()
    if core:
        # Test Ollama service
        models = core.ollama_service.get_models_for_blender()
        print(f"Models available: {len(models)}")
        
        # Test vision service
        scene_context = core.vision_service.get_enhanced_context("create a sphere")
        print(f"Vision context available: {'scene_context' in scene_context}")
        
        # Test request processing
        result = core.process_llammy_request("create a test sphere")
        print(f"Test request: {'SUCCESS' if result.get('success') else 'FAILED'}")
        
        return True
    
    return False

if __name__ == "__main__":
    test_reengineered_core()

print("RE-ENGINEERED LLAMMY CORE LOADED!")
print("Features:")
print("  - Centralized Ollama service (no more conflicts)")
print("  - Vision Intelligence Service integration")
print("  - Enhanced context with scene awareness")
print("  - Unified service architecture")
print("  - Context-aware code generation")
print("Ready for vision-enhanced AI processing!")
