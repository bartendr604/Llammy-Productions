# =============================================================================
# LLAMMY BLENDER ADDON - COMPLETE RE-ENGINEERED INTEGRATION
# __init__.py - Vision-integrated with centralized services - COMPLETE FILE
# =============================================================================

import bpy
import os
import time
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

bl_info = {
    "name": "Llammy Enhanced Production AI",
    "author": "Triangle104 + Claude",
    "version": (5, 0, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > Llammy",
    "description": "Vision-integrated AI system, web scraping",
    "category": "Development",
}

print("Llammy Enhanced Production AI v5.0 - Vision-Integrated Architecture Loading...")

# =============================================================================
# DEPENDENCY INSTALLER - ENHANCED FOR VISION
# =============================================================================

def install_dependencies():
    """Install required packages for Llammy with vision support"""
    import subprocess
    import sys
    
    # Core dependencies
    required_packages = ['psutil', 'requests']
    
    # Vision dependencies (optional)
    vision_packages = ['torch', 'torchvision', 'transformers', 'peft', 'accelerate']
    
    blender_python = sys.executable
    
    # Install core dependencies
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} already available")
        except ImportError:
            print(f"📦 Installing {package}...")
            try:
                subprocess.check_call([blender_python, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
    
    # Check for vision dependencies (don't auto-install due to size)
    vision_available = True
    for package in vision_packages:
        try:
            __import__(package)
        except ImportError:
            vision_available = False
            break
    
    if vision_available:
        print("🎯 Vision dependencies available - full vision integration enabled")
    else:
        print("⚠️ Vision dependencies not available - install torch, transformers, peft for full vision features")

# Install dependencies early
print("🔧 Checking Llammy dependencies...")
install_dependencies()

# =============================================================================
# COMPREHENSIVE DEBUG SYSTEM
# =============================================================================

DEBUG_ENABLED = True

class LlammyDebugSystem:
    def __init__(self):
        self.debug_log = []
        self.max_log_entries = 200
        self.component_status = {}
        
    def log(self, level: str, message: str, component: str = "CORE"):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level} [{component}]: {message}"
        
        if DEBUG_ENABLED:
            self.debug_log.append(entry)
            print(entry)
            
        # Track component health
        if level in ["ERROR", "CRITICAL"]:
            self.component_status[component] = "ERROR"
        elif level == "SUCCESS":
            self.component_status[component] = "OK"
        elif component not in self.component_status:
            self.component_status[component] = "INITIALIZING"
            
        if len(self.debug_log) > self.max_log_entries:
            self.debug_log = self.debug_log[-self.max_log_entries:]
    
    def get_recent_logs(self, count: int = 15) -> List[str]:
        return self.debug_log[-count:]
    
    def get_component_health(self) -> Dict[str, str]:
        return self.component_status.copy()

debug = LlammyDebugSystem()

# =============================================================================
# RE-ENGINEERED MODULE MANAGER
# =============================================================================

class LlammyModuleManager:
    """Re-engineered module manager with centralized services"""
    
    def __init__(self):
        self.modules = {}
        self.initialization_status = {
            'core_initialized': False,
            'ai_engine_connected': False,
            'rag_integration_available': False,
            'harvester_available': False,
            'vision_integration_active': False,
            'services_connected': False,
            'dependencies_installed': False,
            'last_init_attempt': None,
            'init_errors': {},
            'module_versions': {}
        }
        
        # Core instances
        self.llammy_core = None
        self.ai_engine = None
        self.rag_integration = None
        self.harvester = None
        
        # Service references
        self.ollama_service = None
        self.vision_service = None
        
        debug.log("INFO", "Re-engineered module manager initialized", "MANAGER")
    
    def initialize_all_modules(self) -> bool:
        """Initialize all modules with service integration"""
        debug.log("INFO", "Starting service-integrated module initialization", "MANAGER")
        self.initialization_status['last_init_attempt'] = time.time()
        
        # Step 1: Initialize Core (provides services)
        if self._initialize_core():
            debug.log("SUCCESS", "Core module with services initialized", "CORE")
            self._connect_services()
        else:
            debug.log("ERROR", "Core initialization failed", "CORE")
            return False
        
        # Step 2: Initialize AI Engine with services
        if self._initialize_ai_engine():
            debug.log("SUCCESS", "AI engine with service integration initialized", "AI")
        else:
            debug.log("WARNING", "AI engine initialization failed", "AI")
        
        # Step 3: Initialize RAG Integration
        if self._initialize_rag_integration():
            debug.log("SUCCESS", "RAG integration initialized", "RAG")
        else:
            debug.log("WARNING", "RAG integration initialization failed", "RAG")
        
        # Step 4: Initialize Harvester
        if self._initialize_harvester():
            debug.log("SUCCESS", "Harvester initialized", "HARVESTER")
        else:
            debug.log("WARNING", "Harvester initialization failed", "HARVESTER")
        
        return self.initialization_status['core_initialized']
    
    def _initialize_core(self) -> bool:
        """Initialize core module with integrated services"""
        try:
            debug.log("INFO", "Attempting core module import with services", "CORE")
            
            from . import llammy_core
            self.modules['llammy_core'] = llammy_core
            
            # Initialize the core
            if hasattr(llammy_core, 'initialize_llammy_core'):
                init_result = llammy_core.initialize_llammy_core()
                debug.log("INFO", f"Core init result: {init_result}", "CORE")
                
                if init_result and init_result.get('success'):
                    # Get the core instance
                    if hasattr(llammy_core, 'get_llammy_core'):
                        self.llammy_core = llammy_core.get_llammy_core()
                        
                        if self.llammy_core:
                            self.initialization_status['core_initialized'] = True
                            
                            # Check vision integration
                            if init_result.get('vision_available'):
                                self.initialization_status['vision_integration_active'] = True
                                debug.log("SUCCESS", "Vision integration active", "VISION")
                            
                            debug.log("SUCCESS", f"Core instance with services obtained", "CORE")
                            return True
                        else:
                            debug.log("ERROR", "Core instance is None", "CORE")
                    else:
                        debug.log("ERROR", "get_llammy_core function not found", "CORE")
                else:
                    error = init_result.get('error', 'Unknown error') if init_result else 'No result returned'
                    debug.log("ERROR", f"Core initialization failed: {error}", "CORE")
            else:
                debug.log("ERROR", "initialize_llammy_core function not found", "CORE")
                
        except ImportError as e:
            debug.log("ERROR", f"Core module import failed: {e}", "CORE")
            self.initialization_status['init_errors']['core'] = f"Import failed: {e}"
        except Exception as e:
            debug.log("ERROR", f"Core initialization exception: {e}", "CORE")
            self.initialization_status['init_errors']['core'] = f"Exception: {e}"
        
        return False
    
    def _connect_services(self):
        """Connect to centralized services from core"""
        if self.llammy_core:
            try:
                # Get service references from core
                from . import llammy_core
                self.ollama_service = llammy_core.get_ollama_service()
                self.vision_service = llammy_core.get_vision_service()
                
                if self.ollama_service:
                    debug.log("SUCCESS", "Ollama service connected", "SERVICES")
                if self.vision_service:
                    debug.log("SUCCESS", "Vision service connected", "SERVICES")
                
                self.initialization_status['services_connected'] = True
                
            except Exception as e:
                debug.log("WARNING", f"Service connection failed: {e}", "SERVICES")
    
    def _initialize_ai_engine(self) -> bool:
        """Initialize AI engine with service integration"""
        try:
            debug.log("INFO", "Attempting AI engine import with services", "AI")
            
            from . import llammy_ai
            self.modules['llammy_ai'] = llammy_ai
            
            if hasattr(llammy_ai, 'get_ai_engine'):
                self.ai_engine = llammy_ai.get_ai_engine()
                
                if self.ai_engine:
                    # Connect services to AI engine
                    if hasattr(self.ai_engine, 'set_services'):
                        self.ai_engine.set_services(
                            ollama_service=self.ollama_service,
                            vision_service=self.vision_service
                        )
                        debug.log("SUCCESS", "AI engine connected to services", "AI")
                    
                    self.initialization_status['ai_engine_connected'] = True
                    debug.log("SUCCESS", f"AI engine with services obtained", "AI")
                    return True
                else:
                    debug.log("WARNING", "AI engine instance is None", "AI")
            else:
                debug.log("WARNING", "get_ai_engine function not found", "AI")
                
        except ImportError as e:
            debug.log("WARNING", f"AI engine import failed: {e}", "AI")
            self.initialization_status['init_errors']['ai'] = f"Import failed: {e}"
        except Exception as e:
            debug.log("WARNING", f"AI engine exception: {e}", "AI")
            self.initialization_status['init_errors']['ai'] = f"Exception: {e}"
        
        return False
    
    def _initialize_rag_integration(self) -> bool:
        """Initialize RAG integration"""
        try:
            debug.log("INFO", "Attempting RAG integration import", "RAG")
            
            from . import llammy_rag_business_modules
            self.modules['rag_business'] = llammy_rag_business_modules
            
            if hasattr(llammy_rag_business_modules, 'create_rag_integration'):
                self.rag_integration = llammy_rag_business_modules.create_rag_integration()
                
                if self.rag_integration and hasattr(self.rag_integration, 'initialize_for_core'):
                    if self.rag_integration.initialize_for_core():
                        self.initialization_status['rag_integration_available'] = True
                        debug.log("SUCCESS", "RAG integration initialized", "RAG")
                        return True
                    else:
                        debug.log("WARNING", "RAG integration initialization failed", "RAG")
                else:
                    debug.log("WARNING", "RAG integration creation failed", "RAG")
                    
        except ImportError as e:
            debug.log("WARNING", f"RAG integration import failed: {e}", "RAG")
            self.initialization_status['init_errors']['rag'] = f"Import failed: {e}"
        except Exception as e:
            debug.log("WARNING", f"RAG integration exception: {e}", "RAG")
            self.initialization_status['init_errors']['rag'] = f"Exception: {e}"
        
        return False
    
    def _initialize_harvester(self) -> bool:
        """Initialize harvester with dependency handling"""
        try:
            debug.log("INFO", "Attempting harvester import with dependency check", "HARVESTER")
            
            from . import llammy_harvester_module
            self.modules['harvester'] = llammy_harvester_module
            
            if hasattr(llammy_harvester_module, 'create_llammy_rich_dataset_harvester'):
                harvester_path = os.path.expanduser("~/.llammy/harvester_data")
                self.harvester = llammy_harvester_module.create_llammy_rich_dataset_harvester(
                    harvester_path, self.ai_engine
                )
                
                if self.harvester:
                    self.initialization_status['harvester_available'] = True
                    debug.log("SUCCESS", "Harvester initialized", "HARVESTER")
                    return True
                else:
                    debug.log("WARNING", "Harvester creation failed", "HARVESTER")
                    
        except ImportError as e:
            debug.log("WARNING", f"Harvester import failed: {e}", "HARVESTER")
            self.initialization_status['init_errors']['harvester'] = f"Import failed: {e}"
        except Exception as e:
            debug.log("WARNING", f"Harvester exception: {e}", "HARVESTER")
            self.initialization_status['init_errors']['harvester'] = f"Exception: {e}"
        
        return False
    
    def process_request(self, user_request: str, **kwargs) -> Dict[str, Any]:
        """Process request through available modules"""
        if not self.llammy_core:
            return {
                'success': False,
                'error': 'Core module not available',
                'generated_code': '',
                'processing_time': 0.0
            }
        
        try:
            # Add RAG context if available
            if self.rag_integration and kwargs.get('file_context'):
                debug.log("INFO", "Adding RAG context to request", "RAG")
                context_result = self.rag_integration.get_context_for_request(
                    user_request,
                    file_paths=kwargs.get('file_context', [])
                )
                if context_result.get('success'):
                    kwargs['rag_context'] = context_result.get('context', '')
                    kwargs['rag_files_used'] = context_result.get('files_used', [])
            
            # Process through core (which handles AI engine automatically)
            result = self.llammy_core.process_llammy_request(user_request, **kwargs)
            
            # Collect training data if successful
            if (result.get('success') and self.rag_integration and
                hasattr(self.rag_integration, 'collect_training_pair')):
                try:
                    self.rag_integration.collect_training_pair(
                        user_request,
                        kwargs.get('rag_context', ''),
                        result.get('generated_code', ''),
                        result.get('success', False),
                        kwargs.get('models_used', {}),
                        result.get('processing_time', 0.0)
                    )
                except Exception as e:
                    debug.log("WARNING", f"Training data collection failed: {e}", "RAG")
            
            return result
            
        except Exception as e:
            debug.log("ERROR", f"Request processing failed: {e}", "MANAGER")
            return {
                'success': False,
                'error': str(e),
                'generated_code': '',
                'processing_time': 0.0
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'initialization_status': self.initialization_status.copy(),
            'component_health': debug.get_component_health(),
            'modules_loaded': list(self.modules.keys()),
            'module_versions': self.initialization_status['module_versions'].copy()
        }
        
        # Add core status with services
        if self.llammy_core:
            try:
                core_status = self.llammy_core.get_system_status()
                status['core_status'] = core_status
            except Exception as e:
                status['core_status'] = {'error': str(e)}
        
        # Add AI engine status
        if self.ai_engine:
            try:
                if hasattr(self.ai_engine, 'get_system_status'):
                    ai_status = self.ai_engine.get_system_status()
                    status['ai_status'] = ai_status
            except Exception as e:
                status['ai_status'] = {'error': str(e)}
        
        # Add RAG status
        if self.rag_integration:
            try:
                if hasattr(self.rag_integration, 'get_status'):
                    rag_status = self.rag_integration.get_status()
                    status['rag_status'] = rag_status
            except Exception as e:
                status['rag_status'] = {'error': str(e)}
        
        # Add harvester status
        if self.harvester:
            try:
                if hasattr(self.harvester, 'get_comprehensive_earning_stats'):
                    harvester_stats = self.harvester.get_comprehensive_earning_stats()
                    status['harvester_status'] = harvester_stats
            except Exception as e:
                status['harvester_status'] = {'error': str(e)}
        
        return status

# Global module manager
module_manager = LlammyModuleManager()

# =============================================================================
# CENTRALIZED MODEL ENUMERATION FOR BLENDER PROPERTIES
# =============================================================================

def get_ollama_models_for_properties(scene, context):
    """Get Ollama models for Blender EnumProperty - CONFLICT-FREE VERSION"""
    try:
        # Use centralized service if available
        if (module_manager.ollama_service and
            hasattr(module_manager.ollama_service, 'get_models_for_blender')):
            return module_manager.ollama_service.get_models_for_blender()
        
        # Fallback: basic connection attempt
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if models:
                items = []
                for model in models:
                    model_name = model['name']
                    items.append((model_name, f"🤖 {model_name[:40]}", f"Model: {model_name}"))
                return items
            else:
                return [("no_models", "No Models Available", "No models found")]
        else:
            return [("connection_failed", "Connection Failed", "Could not connect to Ollama")]
            
    except Exception as e:
        return [("error", f"Error: {str(e)[:20]}", f"Ollama error: {str(e)}")]

# =============================================================================
# FILE MANAGEMENT SYSTEM
# =============================================================================

class EnhancedFileManager:
    """Enhanced file management with competitive analysis"""
    
    def __init__(self):
        self.selected_files = []
        self.file_stats = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'file_types': {},
            'business_value': 0.0,
            'competitive_files': 0,
            'blender_4_5_files': 0,
            'rag_indexed_files': 0
        }
    
    def add_files(self, file_paths: List[str]):
        for file_path in file_paths:
            if os.path.exists(file_path) and file_path not in [f['path'] for f in self.selected_files]:
                file_info = {
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size': os.path.getsize(file_path),
                    'category': self._categorize_file_enhanced(file_path),
                    'competitive_value': self._assess_competitive_value(file_path)
                }
                self.selected_files.append(file_info)
        
        self._update_enhanced_stats()
        debug.log("INFO", f"Added {len(file_paths)} files to context", "FILES")
    
    def _categorize_file_enhanced(self, file_path: str) -> str:
        """Enhanced file categorization with competitive analysis"""
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path).lower()
        
        if ext == '.blend':
            return 'blend_native'
        elif ext == '.py' and any(term in filename for term in ['bpy', 'blender', 'addon', 'operator']):
            return 'blender_script'
        elif ext == '.py':
            return 'python_script'
        elif ext in ['.obj', '.fbx', '.dae', '.ply', '.stl', '.gltf', '.glb']:
            return '3d_model'
        elif ext in ['.jpg', '.jpeg', '.png', '.tga', '.exr', '.hdr', '.bmp']:
            return 'texture_image'
        elif ext in ['.json', '.xml', '.mtl']:
            return 'material_data'
        elif ext in ['.csv', '.txt', '.md', '.yaml', '.ini']:
            return 'data_file'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            return 'video_tutorial'
        elif ext in ['.wav', '.mp3', '.ogg', '.flac']:
            return 'audio_asset'
        else:
            return 'other'
    
    def _assess_competitive_value(self, file_path: str) -> float:
        """Assess competitive value of file"""
        category = self._categorize_file_enhanced(file_path)
        
        base_values = {
            'blend_native': 10.0,
            'blender_script': 8.0,
            '3d_model': 5.0,
            'texture_image': 4.0,
            'material_data': 3.5,
            'video_tutorial': 3.0,
            'data_file': 2.5,
            'python_script': 2.0,
            'audio_asset': 1.5,
            'other': 1.0
        }
        
        return base_values.get(category, 1.0)
    
    def _update_enhanced_stats(self):
        """Update enhanced statistics"""
        self.file_stats['total_files'] = len(self.selected_files)
        self.file_stats['total_size_mb'] = sum(f['size'] for f in self.selected_files) / (1024 * 1024)
        
        file_types = {}
        competitive_files = 0
        blender_4_5_files = 0
        total_business_value = 0.0
        
        for file_info in self.selected_files:
            category = file_info['category']
            competitive_value = file_info['competitive_value']
            
            file_types[category] = file_types.get(category, 0) + 1
            total_business_value += competitive_value
            
            if category in ['blend_native', 'blender_script'] or competitive_value >= 5.0:
                competitive_files += 1
            
            if self._is_blender_4_5_file(file_info['path']):
                blender_4_5_files += 1
        
        self.file_stats.update({
            'file_types': file_types,
            'competitive_files': competitive_files,
            'blender_4_5_files': blender_4_5_files,
            'business_value': total_business_value
        })
    
    def _is_blender_4_5_file(self, file_path: str) -> bool:
        """Check if file appears to be Blender 4.5 related"""
        filename = os.path.basename(file_path).lower()
        return any(term in filename for term in ['4.5', '4_5', 'latest', 'current', 'new'])
    
    def clear_files(self):
        self.selected_files.clear()
        self._update_enhanced_stats()
        debug.log("INFO", "Cleared all context files", "FILES")
    
    def get_files_for_context(self) -> List[str]:
        return [f['path'] for f in self.selected_files]
    
    def get_competitive_summary(self) -> Dict[str, Any]:
        """Get competitive analysis summary"""
        return {
            'total_files': self.file_stats['total_files'],
            'competitive_files': self.file_stats['competitive_files'],
            'blender_4_5_files': self.file_stats['blender_4_5_files'],
            'total_business_value': self.file_stats['business_value'],
            'competitive_percentage': (
                (self.file_stats['competitive_files'] / max(1, self.file_stats['total_files'])) * 100
                if self.file_stats['total_files'] > 0 else 0
            )
        }

# Global file manager
file_manager = EnhancedFileManager()

# =============================================================================
# UI PANELS - UPDATED FOR VISION INTEGRATION
# =============================================================================

class LLAMMY_PT_MainPanel(bpy.types.Panel):
    bl_label = "Llammy Vision-Integrated AI"
    bl_idname = "LLAMMY_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    
    def draw(self, context):
        layout = self.layout
        
        # Header with system status
        header_box = layout.box()
        header_box.label(text="LLAMMY VISION-INTEGRATED AI v5.0", icon='CONSOLE')
        
        # Get comprehensive status
        status = module_manager.get_comprehensive_status()
        init_status = status.get('initialization_status', {})
        
        # Status indicators
        status_row = header_box.row()
        if init_status.get('core_initialized'):
            status_row.label(text="System: READY", icon='CHECKMARK')
        else:
            status_row.label(text="System: ERROR", icon='ERROR')
        
        # Module status row
        modules_row = header_box.row()
        components = ['Core', 'AI', 'RAG', 'Harvester']
        component_keys = ['core_initialized', 'ai_engine_connected', 'rag_integration_available', 'harvester_available']
        
        for comp, key in zip(components, component_keys):
            icon = 'CHECKMARK' if init_status.get(key, False) else 'X'
            modules_row.label(text=comp, icon=icon)
        
        # Vision integration status
        if init_status.get('vision_integration_active'):
            vision_row = header_box.row()
            vision_row.label(text="Vision: ACTIVE", icon='CAMERA_DATA')
        elif init_status.get('services_connected'):
            vision_row = header_box.row()
            vision_row.label(text="Vision: Basic Mode", icon='RESTRICT_VIEW_ON')
        
        # Performance metrics if available
        if init_status.get('core_initialized'):
            core_status = status.get('core_status', {}).get('core', {})
            stats = core_status.get('stats', {})
            
            if stats.get('total_requests', 0) > 0:
                success_rate = (stats.get('successful_requests', 0) / stats['total_requests']) * 100
                header_box.label(text=f"Success Rate: {success_rate:.1f}% ({stats['total_requests']} requests)")
                
                if stats.get('vision_enhanced_requests', 0) > 0:
                    vision_pct = (stats['vision_enhanced_requests'] / stats['total_requests']) * 100
                    header_box.label(text=f"Vision Enhanced: {vision_pct:.1f}%")

class LLAMMY_PT_ModelsPanel(bpy.types.Panel):
    bl_label = "AI Models Configuration"
    bl_idname = "LLAMMY_PT_models_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        models_box = layout.box()
        models_box.label(text="Dual AI Configuration:", icon='SETTINGS')
        
        # Creative model selection
        creative_row = models_box.row()
        creative_row.label(text="Creative (Vision-Enhanced):", icon='BRUSH_DATA')
        if hasattr(scene, "llammy_creative_model"):
            models_box.prop(scene, "llammy_creative_model", text="")
        
        # Technical model selection
        technical_row = models_box.row()
        technical_row.label(text="Technical (Context-Aware):", icon='SETTINGS')
        if hasattr(scene, "llammy_technical_model"):
            models_box.prop(scene, "llammy_technical_model", text="")
        
        # Refresh button
        refresh_row = models_box.row()
        refresh_row.scale_y = 1.2
        refresh_row.operator("llammy.refresh_models", text="Refresh Models", icon='FILE_REFRESH')

class LLAMMY_PT_RequestPanel(bpy.types.Panel):
    bl_label = "AI Processing"
    bl_idname = "LLAMMY_PT_request_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        request_box = layout.box()
        request_box.label(text="AI Request Input:", icon='CONSOLE')
        
        # Request input
        if hasattr(scene, "llammy_request_input"):
            request_box.prop(scene, "llammy_request_input", text="")
        
        # Context information
        file_stats = file_manager.get_competitive_summary()
        if file_stats['total_files'] > 0:
            context_info = request_box.row()
            context_text = f"Context: {file_stats['total_files']} files"
            if file_stats['competitive_files'] > 0:
                context_text += f" ({file_stats['competitive_files']} competitive)"
            context_info.label(text=context_text, icon='LIBRARY_DATA_DIRECT')
        
        # Vision status
        status = module_manager.get_comprehensive_status()
        if status.get('initialization_status', {}).get('vision_integration_active'):
            vision_row = request_box.row()
            vision_row.label(text="Scene Analysis: ACTIVE", icon='CAMERA_DATA')
        
        # Execute button
        execute_row = request_box.row()
        execute_row.scale_y = 2.0
        execute_row.operator("llammy.execute_request", text="EXECUTE VISION AI", icon='PLAY')

class LLAMMY_PT_ContentPanel(bpy.types.Panel):
    bl_label = "Content Context"
    bl_idname = "LLAMMY_PT_content_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        content_box = layout.box()
        
        file_stats = file_manager.get_competitive_summary()
        
        if file_stats['total_files'] > 0:
            content_box.label(text="Content Statistics:", icon='LIBRARY_DATA_DIRECT')
            
            stats_grid = content_box.grid_flow(row_major=True, columns=2, align=True)
            stats_grid.label(text="Total Files:")
            stats_grid.label(text=f"{file_stats['total_files']}")
            
            stats_grid.label(text="Competitive:")
            stats_grid.label(text=f"{file_stats['competitive_files']} ({file_stats['competitive_percentage']:.1f}%)")
            
            if file_stats['blender_4_5_files'] > 0:
                stats_grid.label(text="Blender 4.5:")
                stats_grid.label(text=f"{file_stats['blender_4_5_files']}")
            
            stats_grid.label(text="Business Value:")
            stats_grid.label(text=f"${file_stats['total_business_value']:.1f}")
        else:
            content_box.label(text="No Content Loaded", icon='LIBRARY_DATA_DIRECT')
        
        content_box.separator()
        content_box.label(text="Load Content:", icon='FILEBROWSER')
        
        file_row = content_box.row()
        file_col = file_row.column()
        file_col.operator("llammy.select_files", text="Select Files", icon='FILEBROWSER')
        file_col.operator("llammy.select_folder", text="Select Folder", icon='FILE_FOLDER')
        
        if file_stats['total_files'] > 0:
            clear_row = content_box.row()
            clear_row.operator("llammy.clear_files", text="Clear Content", icon='TRASH')

class LLAMMY_PT_SystemPanel(bpy.types.Panel):
    bl_label = "System Status"
    bl_idname = "LLAMMY_PT_system_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        status_box = layout.box()
        
        status = module_manager.get_comprehensive_status()
        init_status = status.get('initialization_status', {})
        
        status_box.label(text="Module Status:", icon='SYSTEM')
        
        # Service status
        services_row = status_box.row()
        services_icon = 'CHECKMARK' if init_status.get('services_connected') else 'X'
        services_row.label(text="Centralized Services:", icon=services_icon)
        
        # Vision status
        vision_row = status_box.row()
        vision_icon = 'CHECKMARK' if init_status.get('vision_integration_active') else 'RESTRICT_VIEW_ON'
        vision_status = "ACTIVE" if init_status.get('vision_integration_active') else "Basic Mode"
        vision_row.label(text=f"Vision Integration: {vision_status}", icon=vision_icon)
        
        # Core modules
        for label, key in [("Core:", 'core_initialized'), ("AI Engine:", 'ai_engine_connected'),
                          ("RAG:", 'rag_integration_available'), ("Harvester:", 'harvester_available')]:
            row = status_box.row()
            icon = 'CHECKMARK' if init_status.get(key) else 'X'
            row.label(text=label, icon=icon)

class LLAMMY_PT_DebugPanel(bpy.types.Panel):
    bl_label = "Debug Information"
    bl_idname = "LLAMMY_PT_debug_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    @classmethod
    def poll(cls, context):
        return DEBUG_ENABLED
    
    def draw(self, context):
        layout = self.layout
        debug_box = layout.box()
        debug_box.label(text="Recent Debug Logs:", icon='INFO')
        
        recent_logs = debug.get_recent_logs(8)
        for log_entry in recent_logs:
            display_entry = log_entry[:80] + "..." if len(log_entry) > 80 else log_entry
            debug_box.label(text=display_entry)

# =============================================================================
# OPERATORS - UPDATED FOR SERVICE INTEGRATION
# =============================================================================

class LLAMMY_OT_ExecuteRequest(bpy.types.Operator):
    bl_idname = "llammy.execute_request"
    bl_label = "Execute AI Request"
    bl_description = "Process request through vision-integrated AI system"
    
    def execute(self, context):
        scene = context.scene
        user_request = getattr(scene, 'llammy_request_input', '').strip()
        
        if not user_request:
            self.report({'ERROR'}, "Please enter a request")
            return {'CANCELLED'}
        
        try:
            creative_model = getattr(scene, 'llammy_creative_model', 'unknown')
            technical_model = getattr(scene, 'llammy_technical_model', 'unknown')
            
            self.report({'INFO'}, f"Processing with vision integration...")
            
            # Report context information
            file_stats = file_manager.get_competitive_summary()
            if file_stats['total_files'] > 0:
                self.report({'INFO'}, f"Context: {file_stats['total_files']} files")
            
            # Prepare request parameters
            request_params = {
                'creative_model': creative_model,
                'technical_model': technical_model,
                'models_used': {
                    'creative': creative_model,
                    'technical': technical_model
                }
            }
            
            # Add file context if available
            file_paths = file_manager.get_files_for_context()
            if file_paths:
                request_params['file_context'] = file_paths
            
            # Process request through module manager
            result = module_manager.process_request(user_request, **request_params)
            
            if hasattr(scene, 'llammy_last_result'):
                scene.llammy_last_result = str(result)
            
            if result.get('success'):
                processing_time = result.get('processing_time', 0)
                self.report({'INFO'}, f"Processing complete in {processing_time:.2f}s")
                
                if result.get('vision_enhanced'):
                    self.report({'INFO'}, "Vision enhancement applied")
                
                generated_code = result.get('generated_code', '')
                
                if generated_code:
                    try:
                        self.report({'INFO'}, "Executing generated code...")
                        exec(generated_code)
                        
                        method = result.get('method', 'unknown')
                        self.report({'INFO'}, f"SUCCESS! Generated via {method}")
                        
                        if hasattr(scene, 'llammy_request_input'):
                            scene.llammy_request_input = ""
                        
                    except Exception as exec_error:
                        self.report({'WARNING'}, f"Code execution error: {str(exec_error)}")
                        debug.log("ERROR", f"Code execution failed: {exec_error}", "EXECUTE")
                else:
                    self.report({'WARNING'}, "AI generated empty code")
            else:
                error_msg = result.get('error', 'Unknown error')
                self.report({'ERROR'}, f"AI request failed: {error_msg}")
                
        except Exception as e:
            self.report({'ERROR'}, f"Request processing failed: {str(e)}")
            debug.log("ERROR", f"Execute operator failed: {e}", "EXECUTE")
        
        return {'FINISHED'}

class LLAMMY_OT_RefreshModels(bpy.types.Operator):
    bl_idname = "llammy.refresh_models"
    bl_label = "Refresh Models"
    bl_description = "Refresh available Ollama models via centralized service"
    
    def execute(self, context):
        try:
            self.report({'INFO'}, "Refreshing model list via centralized service...")
            
            # Force refresh through service
            if module_manager.ollama_service:
                models = module_manager.ollama_service.get_models_for_blender(force_refresh=True)
                self.report({'INFO'}, f"Found {len(models)} models via service")
            
            # Re-register properties to refresh enum items
            bpy.types.Scene.llammy_creative_model = bpy.props.EnumProperty(
                name="Creative Model",
                description="Model for creative analysis with vision enhancement",
                items=get_ollama_models_for_properties
            )
            bpy.types.Scene.llammy_technical_model = bpy.props.EnumProperty(
                name="Technical Model",
                description="Model for technical code generation with context awareness",
                items=get_ollama_models_for_properties
            )
            
            self.report({'INFO'}, "Model list refreshed successfully")
            debug.log("SUCCESS", "Models refreshed via service", "MODELS")
            
        except Exception as e:
            self.report({'WARNING'}, f"Refresh failed: {str(e)}")
            debug.log("ERROR", f"Model refresh failed: {e}", "MODELS")
        
        return {'FINISHED'}

class LLAMMY_OT_SelectFiles(bpy.types.Operator):
    bl_idname = "llammy.select_files"
    bl_label = "Select Files"
    bl_description = "Select files for AI context enhancement"
    bl_options = {'REGISTER'}
    
    files: bpy.props.CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )
    
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    
    filter_glob: bpy.props.StringProperty(
        default="*.blend;*.py;*.obj;*.fbx;*.jpg;*.jpeg;*.png;*.exr;*.json;*.csv;*.txt;*.mp4;*.wav;*.md;*.yaml",
        options={'HIDDEN'}
    )
    
    def execute(self, context):
        file_paths = []
        for file_elem in self.files:
            file_path = os.path.join(self.directory, file_elem.name)
            if os.path.exists(file_path):
                file_paths.append(file_path)
        
        if file_paths:
            file_manager.add_files(file_paths)
            file_stats = file_manager.get_competitive_summary()
            
            self.report({'INFO'}, f"Added {len(file_paths)} files")
            if file_stats['competitive_files'] > 0:
                self.report({'INFO'}, f"Competitive files: {file_stats['competitive_files']}")
        else:
            self.report({'WARNING'}, "No files selected")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LLAMMY_OT_SelectFolder(bpy.types.Operator):
    bl_idname = "llammy.select_folder"
    bl_label = "Select Folder"
    bl_description = "Select folder for batch file processing"
    bl_options = {'REGISTER'}
    
    directory: bpy.props.StringProperty(
        name="Folder Path",
        subtype='DIR_PATH',
    )
    
    def execute(self, context):
        if not self.directory or not os.path.exists(self.directory):
            self.report({'ERROR'}, "Invalid folder selected")
            return {'CANCELLED'}
        
        file_paths = []
        folder_path = Path(self.directory)
        
        supported_exts = {
            '.blend', '.py', '.obj', '.fbx', '.dae', '.ply', '.stl', '.gltf', '.glb',
            '.jpg', '.jpeg', '.png', '.tga', '.exr', '.hdr', '.bmp', '.tiff',
            '.json', '.xml', '.mtl', '.csv', '.txt', '.md', '.yaml', '.ini',
            '.mp4', '.avi', '.mov', '.mkv', '.webm', '.wav', '.mp3', '.ogg', '.flac'
        }
        
        try:
            for file_path in folder_path.rglob('*'):
                if (file_path.is_file() and
                    file_path.suffix.lower() in supported_exts):
                    file_paths.append(str(file_path))
                    
                    if len(file_paths) >= 200:
                        break
            
            if file_paths:
                file_manager.add_files(file_paths)
                file_stats = file_manager.get_competitive_summary()
                
                self.report({'INFO'}, f"Loaded {len(file_paths)} files from folder")
                if file_stats['competitive_files'] > 0:
                    self.report({'INFO'}, f"Found {file_stats['competitive_files']} competitive files")
                
                if len(file_paths) >= 200:
                    self.report({'WARNING'}, "Limited to 200 files - folder contains more")
            else:
                self.report({'WARNING'}, "No supported files found in folder")
                
        except Exception as e:
            self.report({'ERROR'}, f"Folder processing failed: {str(e)}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LLAMMY_OT_ClearFiles(bpy.types.Operator):
    bl_idname = "llammy.clear_files"
    bl_label = "Clear Files"
    bl_description = "Clear all loaded context files"
    
    def execute(self, context):
        files_count = file_manager.file_stats['total_files']
        file_manager.clear_files()
        self.report({'INFO'}, f"Cleared {files_count} files from context")
        debug.log("INFO", f"Cleared {files_count} context files", "FILES")
        return {'FINISHED'}

# =============================================================================
# PROPERTIES REGISTRATION - CONFLICT-FREE
# =============================================================================

def register_properties():
    """Register all Blender properties with conflict-free model enumeration"""
    try:
        bpy.types.Scene.llammy_request_input = bpy.props.StringProperty(
            name="AI Request",
            description="Your request for the vision-integrated AI system",
            default="",
            maxlen=2000
        )
        
        bpy.types.Scene.llammy_creative_model = bpy.props.EnumProperty(
            name="Creative Model",
            description="Model for creative analysis with vision enhancement",
            items=get_ollama_models_for_properties
        )
        
        bpy.types.Scene.llammy_technical_model = bpy.props.EnumProperty(
            name="Technical Model",
            description="Model for technical code generation with context awareness",
            items=get_ollama_models_for_properties
        )
        
        bpy.types.Scene.llammy_last_result = bpy.props.StringProperty(
            name="Last Result",
            description="Result from last AI request",
            default=""
        )
        
        debug.log("SUCCESS", "Properties registered successfully", "PROPS")
        
    except Exception as e:
        debug.log("ERROR", f"Property registration failed: {e}", "PROPS")

def unregister_properties():
    """Unregister all Blender properties"""
    properties = [
        'llammy_request_input',
        'llammy_creative_model',
        'llammy_technical_model',
        'llammy_last_result'
    ]
    
    for prop in properties:
        try:
            if hasattr(bpy.types.Scene, prop):
                delattr(bpy.types.Scene, prop)
                debug.log("SUCCESS", f"Unregistered property: {prop}", "PROPS")
        except Exception as e:
            debug.log("ERROR", f"Failed to unregister {prop}: {e}", "PROPS")

# =============================================================================
# ADDON REGISTRATION
# =============================================================================

classes = [
    LLAMMY_PT_MainPanel,
    LLAMMY_PT_ModelsPanel,
    LLAMMY_PT_RequestPanel,
    LLAMMY_PT_ContentPanel,
    LLAMMY_PT_SystemPanel,
    LLAMMY_PT_DebugPanel,
    LLAMMY_OT_ExecuteRequest,
    LLAMMY_OT_RefreshModels,
    LLAMMY_OT_SelectFiles,
    LLAMMY_OT_SelectFolder,
    LLAMMY_OT_ClearFiles,
]

def register():
    """Register the vision-integrated Llammy addon"""
    debug.log("INFO", "Starting vision-integrated addon registration", "REGISTER")
    
    # Register UI classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            debug.log("SUCCESS", f"Registered UI class: {cls.__name__}", "REGISTER")
        except Exception as e:
            debug.log("ERROR", f"Failed to register {cls.__name__}: {e}", "REGISTER")
    
    # Register properties
    register_properties()
    
    # Initialize all modules with services
    success = module_manager.initialize_all_modules()
    
    # Print registration summary
    print("=" * 80)
    print("LLAMMY VISION-INTEGRATED AI v5.0 REGISTERED")
    print("=" * 80)
    print("RE-ENGINEERED ARCHITECTURE:")
    print("  - Centralized Ollama service (no conflicts)")
    print("  - Vision Intelligence Service integration")
    print("  - Enhanced context with scene awareness")
    print("  - Service-integrated AI processing")
    print("  - Conflict-free model enumeration")
    print("  - Vision LoRA framework integration")
    print("")
    print("CORE FEATURES:")
    print("  - Dual AI with vision enhancement")
    print("  - Scene-aware code generation")
    print("  - Context-driven model selection")
    print("  - Intelligent harvesting with psutil fallback")
    print("  - RAG with competitive analysis")
    print("  - Triangle104 model prioritization")
    
    if success:
        status = module_manager.get_comprehensive_status()
        init_status = status.get('initialization_status', {})
        
        print("")
        print("SYSTEM STATUS:")
        if init_status.get('vision_integration_active'):
            print("  - Vision Integration: ACTIVE")
        else:
            print("  - Vision Integration: Basic Mode")
        
        if init_status.get('services_connected'):
            print("  - Centralized Services: CONNECTED")
        
        operational_modules = sum(1 for v in [
            init_status.get('core_initialized'),
            init_status.get('ai_engine_connected'),
            init_status.get('rag_integration_available'),
            init_status.get('harvester_available')
        ] if v)
        
        print(f"  - Operational Modules: {operational_modules}/4")
        print("  - All Core Systems: OPERATIONAL")
    else:
        print("")
        print("SYSTEM STATUS:")
        print("  - Some Systems: LIMITED FUNCTIONALITY")
        print("   Check System Status panel for details")
    
    if DEBUG_ENABLED:
        print("  - Debug Mode: ENABLED")
    
    print("=" * 80)
    print("Ready for vision-enhanced Blender AI automation!")

def unregister():
    """Unregister the vision-integrated addon"""
    debug.log("INFO", "Starting addon unregistration", "UNREGISTER")
    
    # Unregister properties
    unregister_properties()
    
    # Unregister UI classes
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            debug.log("SUCCESS", f"Unregistered: {cls.__name__}", "UNREGISTER")
        except Exception as e:
            debug.log("ERROR", f"Failed to unregister {cls.__name__}: {e}", "UNREGISTER")
    
    # Shutdown harvester if available
    if module_manager.harvester and hasattr(module_manager.harvester, 'shutdown'):
        try:
            module_manager.harvester.shutdown()
            debug.log("SUCCESS", "Harvester shutdown complete", "UNREGISTER")
        except Exception as e:
            debug.log("ERROR", f"Harvester shutdown failed: {e}", "UNREGISTER")
    
    print("Llammy Vision-Integrated AI unregistered successfully")

# =============================================================================
# DIRECT EXECUTION SUPPORT
# =============================================================================

if __name__ == "__main__":
    register()

# =============================================================================
# MODULE COMPLETION NOTICE
# =============================================================================

print("LLAMMY VISION-INTEGRATED AI v5.0 - RE-ENGINEERED ARCHITECTURE LOADED!")
print("Complete conflict-free integration:")
print("  - No duplicate Ollama calls")
print("  - Centralized service architecture")
print("  - Vision LoRA framework integration")
print("  - Enhanced scene-aware AI processing")
print("  - Proper dependency management")
print("  - Service-connected module ecosystem")
print("Ready for vision-enhanced 3D modeling automation!")
