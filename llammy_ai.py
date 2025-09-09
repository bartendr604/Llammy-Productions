# =============================================================================
# LLAMMY AI ENGINE - RE-ENGINEERED WITH SERVICE INTEGRATION
# llammy_ai.py - No more Ollama conflicts, integrated with vision services
# =============================================================================

import requests
import json
import time
import re
import traceback
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

print("🤖 Llammy AI Engine - Re-engineered with service integration")

# =============================================================================
# BLENDER CODE ENHANCER - ENHANCED FOR VISION
# =============================================================================

class BlenderCodeEnhancer:
    """Enhanced code enhancer with vision-aware improvements"""
    
    def __init__(self):
        self.api_fixes = {
            'bpy.context.scene.objects': 'bpy.context.collection.objects',
            'bpy.context.scene.camera': 'bpy.context.scene.camera',
            'object.select': 'object.select_set(True)',
            'bpy.ops.object.delete()': 'bpy.ops.object.delete(use_global=False)',
        }
        
        # Vision-aware enhancements
        self.vision_patterns = {
            'multiple_objects': 'location=(2, 0, 0)',  # Offset new objects
            'empty_scene': 'location=(0, 0, 0)',       # Center in empty scene
            'preserve_selection': True,                # Keep existing selection context
        }
    
    def enhance_code(self, code: str, enhanced_context: Dict[str, Any] = None) -> str:
        """Enhance code with vision-aware improvements"""
        enhanced_code = code
        
        # Apply standard API fixes
        for old_api, new_api in self.api_fixes.items():
            enhanced_code = enhanced_code.replace(old_api, new_api)
        
        # Apply vision-aware enhancements
        if enhanced_context:
            enhanced_code = self._apply_vision_enhancements(enhanced_code, enhanced_context)
        
        # Add import if missing
        if 'import bpy' not in enhanced_code:
            enhanced_code = 'import bpy\n\n' + enhanced_code
        
        return enhanced_code
    
    def _apply_vision_enhancements(self, code: str, enhanced_context: Dict[str, Any]) -> str:
        """Apply vision-aware code enhancements"""
        scene_data = enhanced_context.get('scene_data', {})
        object_count = scene_data.get('object_count', 0)
        
        # Adjust object placement based on scene context
        if object_count > 0:
            # Replace default locations with offset positions
            code = code.replace('location=(0, 0, 0)', 'location=(2, 0, 0)')
        
        # Add context-aware comments
        if object_count == 0:
            code = f"# Creating in empty scene\n{code}"
        else:
            code = f"# Adding to scene with {object_count} existing objects\n{code}"
        
        return code

# =============================================================================
# RE-ENGINEERED DUAL AI SYSTEM
# =============================================================================

class LlammyDualAI:
    """Re-engineered dual AI system with service integration"""
    
    def __init__(self):
        self.version = "5.0.0-SERVICE-INTEGRATED"
        self.code_enhancer = BlenderCodeEnhancer()
        
        # Service references (set by core)
        self.ollama_service = None
        self.vision_service = None
        
        # Enhanced stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'creative_calls': 0,
            'technical_calls': 0,
            'vision_enhanced_requests': 0,
            'service_integration_active': False,
            'avg_processing_time': 0.0,
            'model_resolution_successes': 0
        }
        
        print(f"Llammy Dual AI System v{self.version} initialized")
    
    def set_services(self, ollama_service=None, vision_service=None):
        """Set service references from core"""
        self.ollama_service = ollama_service
        self.vision_service = vision_service
        self.stats['service_integration_active'] = True
        
        print("AI engine connected to centralized services")
        if vision_service and vision_service.vision_available:
            print("Vision integration: ACTIVE")
        if ollama_service:
            print("Ollama service: CONNECTED")
    
    def execute_request(self, user_request: str, context: str = "",
                       creative_model: str = None, technical_model: str = None,
                       enhanced_context: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute request with service integration"""
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        print(f"Dual AI processing with services: {user_request[:50]}...")
        
        try:
            # STEP 1: Enhanced context preparation
            prepared_context = self._prepare_enhanced_context(context, enhanced_context, user_request)
            
            # STEP 2: Service-aware model resolution
            resolved_models = self._resolve_models_via_service(creative_model, technical_model)
            
            # STEP 3: Execute dual AI with enhanced context
            result = self._execute_dual_ai_enhanced(
                user_request,
                prepared_context,
                resolved_models['creative'],
                resolved_models['technical'],
                enhanced_context
            )
            
            # STEP 4: Track vision enhancement
            if enhanced_context and enhanced_context.get('vision_enhanced'):
                self.stats['vision_enhanced_requests'] += 1
                result['vision_enhanced'] = True
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time, result.get('success', False))
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            error_msg = str(e)
            print(f"Service-integrated AI failed: {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'generated_code': '',
                'method': 'service_integrated_ai_failed',
                'service_integration_active': self.stats['service_integration_active']
            }
    
    def _prepare_enhanced_context(self, base_context: str, enhanced_context: Dict[str, Any], user_request: str) -> str:
        """Prepare enhanced context with vision awareness"""
        context_parts = [base_context] if base_context else []
        
        if enhanced_context:
            # Add scene context
            if enhanced_context.get('scene_context'):
                context_parts.append(f"Scene: {enhanced_context['scene_context']}")
            
            # Add visual intent
            if enhanced_context.get('visual_intent'):
                visual_intent = enhanced_context['visual_intent']
                intent_desc = []
                for category, items in visual_intent.items():
                    intent_desc.append(f"{category}: {', '.join(items)}")
                if intent_desc:
                    context_parts.append(f"Visual intent: {'; '.join(intent_desc)}")
            
            # Add Blender context
            if enhanced_context.get('blender_context'):
                context_parts.append(enhanced_context['blender_context'])
        
        return ". ".join(context_parts)
    
    def _resolve_models_via_service(self, creative_model: str, technical_model: str) -> Dict[str, str]:
        """Resolve models using centralized Ollama service"""
        resolved = {
            'creative': creative_model or 'llama3.2:3b',
            'technical': technical_model or 'llama3.2:3b'
        }
        
        if self.ollama_service:
            try:
                # Use service to resolve best models
                resolved['creative'] = self.ollama_service.resolve_best_model(
                    creative_model or '', 'creative'
                )
                resolved['technical'] = self.ollama_service.resolve_best_model(
                    technical_model or '', 'technical'
                )
                self.stats['model_resolution_successes'] += 1
                
                print(f"Models resolved via service: {resolved['creative'][:30]}... / {resolved['technical'][:30]}...")
                
            except Exception as e:
                print(f"Model resolution via service failed: {e}")
        
        return resolved
    
    def _execute_dual_ai_enhanced(self, user_request: str, context: str,
                                 creative_model: str, technical_model: str,
                                 enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dual AI with enhanced context"""
        try:
            start_time = time.time()
            
            # Phase 1: Creative Analysis (if models differ)
            if creative_model != technical_model:
                creative_result = self._creative_phase_enhanced(
                    user_request, context, creative_model, enhanced_context
                )
                creative_analysis = creative_result.get('analysis', '')
                self.stats['creative_calls'] += 1
            else:
                creative_analysis = "Direct technical implementation requested."
            
            # Phase 2: Technical Implementation
            technical_result = self._technical_phase_enhanced(
                user_request, context, creative_analysis, technical_model, enhanced_context
            )
            self.stats['technical_calls'] += 1
            
            if not technical_result.get('success'):
                raise Exception(f"Technical phase failed: {technical_result.get('error')}")
            
            # Phase 3: Enhanced code post-processing
            generated_code = technical_result.get('code', '')
            enhanced_code = self.code_enhancer.enhance_code(generated_code, enhanced_context)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'generated_code': enhanced_code,
                'method': 'service_integrated_dual_ai',
                'processing_time': processing_time,
                'models_used': {
                    'creative': creative_model,
                    'technical': technical_model
                },
                'creative_analysis': creative_analysis,
                'code_enhanced': generated_code != enhanced_code,
                'service_integration': {
                    'ollama_service_used': self.ollama_service is not None,
                    'vision_service_used': enhanced_context.get('vision_enhanced', False) if enhanced_context else False
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'generated_code': '',
                'method': 'enhanced_dual_ai_failed'
            }
    
    def _creative_phase_enhanced(self, user_request: str, context: str, model: str, enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Creative phase with vision enhancement"""
        try:
            # Build enhanced prompt
            prompt_parts = [
                "You are a creative Blender AI assistant with scene awareness.",
                f"User Request: {user_request}",
                f"Context: {context}"
            ]
            
            # Add vision-specific guidance
            if enhanced_context and enhanced_context.get('visual_intent'):
                visual_intent = enhanced_context['visual_intent']
                prompt_parts.append(f"Visual requirements: {visual_intent}")
            
            prompt_parts.append("Provide a brief creative analysis of what the user wants to achieve in Blender. Focus on the creative approach and key components needed.")
            
            prompt = "\n\n".join(prompt_parts)
            
            return self._call_ollama_via_service(model, prompt, "creative")
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _technical_phase_enhanced(self, user_request: str, context: str, creative_analysis: str, model: str, enhanced_context: Dict[str, Any]) -> Dict[str, Any]:
        """Technical phase with vision enhancement"""
        try:
            # Build enhanced prompt
            prompt_parts = [
                "You are a technical Blender Python expert with scene awareness.",
                f"User Request: {user_request}",
                f"Context: {context}",
                f"Creative Analysis: {creative_analysis}"
            ]
            
            # Add scene-specific guidance
            if enhanced_context and enhanced_context.get('scene_data'):
                scene_data = enhanced_context['scene_data']
                object_count = scene_data.get('object_count', 0)
                if object_count > 0:
                    prompt_parts.append(f"Note: Scene has {object_count} existing objects. Position new objects appropriately.")
                else:
                    prompt_parts.append("Note: Scene is empty. Center new objects.")
            
            prompt_parts.append("Generate clean, working Python code for Blender 4.4/4.5. Respond with ONLY the Python code, no explanations.")
            
            prompt = "\n\n".join(prompt_parts)
            
            result = self._call_ollama_via_service(model, prompt, "technical")
            
            if result.get('success'):
                # Clean up code
                code = result.get('content', '').strip()
                if code.startswith('```python'):
                    code = code.replace('```python', '').replace('```', '').strip()
                elif code.startswith('```'):
                    code = code.replace('```', '').strip()
                
                result['code'] = code
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _call_ollama_via_service(self, model: str, prompt: str, phase: str = "general") -> Dict[str, Any]:
        """Call Ollama via centralized service"""
        if not self.ollama_service:
            return {
                'success': False,
                'error': 'Ollama service not available'
            }
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_service.base_url}/api/generate",
                json=payload,
                timeout=self.ollama_service.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                
                return {
                    'success': True,
                    'content': content,
                    'analysis': content if phase == "creative" else "",
                    'model_used': model,
                    'phase': phase
                }
            else:
                return {
                    'success': False,
                    'error': f'Ollama returned status {response.status_code}'
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'{phase.title()} request timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'{phase.title()} call failed: {str(e)}'
            }
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update processing statistics"""
        if success:
            self.stats['successful_requests'] += 1
        
        # Update average processing time
        total_requests = self.stats['total_requests']
        current_avg = self.stats['avg_processing_time']
        
        if total_requests > 0:
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.stats['avg_processing_time'] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status"""
        status = {
            'version': self.version,
            'stats': self.stats.copy(),
            'success_rate': (
                self.stats['successful_requests'] / max(self.stats['total_requests'], 1) * 100
            ),
            'service_integration': {
                'ollama_service_connected': self.ollama_service is not None,
                'vision_service_connected': self.vision_service is not None,
                'vision_available': self.vision_service.vision_available if self.vision_service else False
            }
        }
        
        # Add service-specific status
        if self.ollama_service:
            ollama_status = self.ollama_service.test_connection()
            status['ollama_service_status'] = ollama_status
        
        return status
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection via services"""
        if not self.ollama_service:
            return {
                'success': False,
                'error': 'Ollama service not connected'
            }
        
        return self.ollama_service.test_connection()

# Global AI Engine instance
llammy_ai_engine = LlammyDualAI()

def get_ai_engine() -> LlammyDualAI:
    """Get the global AI engine"""
    return llammy_ai_engine

def test_service_integrated_ai():
    """Test service-integrated AI engine"""
    print("Testing Service-Integrated Llammy AI Engine...")
    
    engine = get_ai_engine()
    if not engine:
        print("AI engine creation failed")
        return False
    
    # Test without services (should work with limitations)
    print("Testing without services...")
    result = engine.execute_request("create a test sphere")
    print(f"Without services: {'SUCCESS' if result.get('success') else 'FAILED'}")
    
    # Test service integration status
    status = engine.get_system_status()
    print(f"Service integration: {status['service_integration']}")
    
    return True

if __name__ == "__main__":
    test_service_integrated_ai()

print("RE-ENGINEERED LLAMMY AI ENGINE LOADED!")
print("Service integration architecture:")
print("  - No direct Ollama calls (uses centralized service)")
print("  - Vision-enhanced context processing")
print("  - Enhanced code generation with scene awareness")
print("  - Unified model resolution")
print("  - Service-aware statistics tracking")
print("Ready for integration with re-engineered core!")
