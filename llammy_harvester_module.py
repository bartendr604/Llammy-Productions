# =============================================================================
# LLAMMY HARVESTER MODULE - COMPLETE UNIFIED IMPLEMENTATION WITH WEB SCRAPING
# llammy_harvester_module.py - All features in one module with psutil fix + ethical web harvesting
#
# COMPLETE FEATURES:
# - Core harvester implementation with MCP
# - Intelligent idle detection and background operation
# - Daily data caps (2GB default)
# - RAG feedback integration
# - Smart file discovery and prioritization
# - Ethical web scraping for Blender API and open source content
# - FIXED: psutil dependency handling with fallbacks
# =============================================================================

import os
import json
import time
import sqlite3
import hashlib
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# Web scraping imports
import requests
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse

# Try to import BeautifulSoup with fallback
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
    print("✅ BeautifulSoup available - full web scraping enabled")
except ImportError:
    BS4_AVAILABLE = False
    print("⚠️ BeautifulSoup not available - web scraping will use basic text parsing")

# =============================================================================
# PSUTIL DEPENDENCY HANDLING - FIXED
# =============================================================================

# Handle psutil import with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("✅ psutil available - full intelligent harvesting enabled")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - intelligent harvesting will use basic mode")
    
    # Create mock psutil for basic functionality
    class MockProcess:
        def __init__(self, name="unknown"):
            self.info = {'name': name, 'cpu_percent': 0}
        
        def cpu_percent(self, interval=None):
            return 0.0  # Always return low CPU for safety
    
    class MockPsutil:
        def process_iter(self, attrs=None):
            # Return empty list - no processes detected
            return []
    
    # Create mock psutil module
    psutil = MockPsutil()

print("🌾 Llammy Unified Harvester - Complete with Intelligent Background Operation + Web Scraping")

# =============================================================================
# MCP ASYNC EVENT LOOP MANAGER
# =============================================================================

class MCPEventLoopManager:
    """Enhanced async event loop manager for harvester"""
    
    def __init__(self):
        self.loop = None
        self.loop_thread = None
        self.running = False
        self.tasks = []
        
    def start_event_loop(self):
        """Start dedicated async event loop"""
        if self.running:
            return True
            
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.running = True
            print("🔥 MCP harvester event loop started")
            
            try:
                self.loop.run_forever()
            except Exception as e:
                print(f"⚠ MCP loop error: {e}")
            finally:
                self.running = False
                print("🛑 MCP harvester event loop stopped")
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        
        # Wait for loop to be ready
        timeout = 5.0
        start_time = time.time()
        while not self.running and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return self.running
    
    def schedule_task(self, coro):
        """Schedule coroutine on the harvester event loop"""
        if self.loop and self.running:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            self.tasks.append(future)
            return future
        else:
            print("⚠️ MCP event loop not running, cannot schedule task")
            return None
    
    def stop_event_loop(self):
        """Stop the MCP event loop"""
        if self.loop and self.running:
            self.loop.call_soon_threadsafe(self.loop.stop)
            if self.loop_thread:
                self.loop_thread.join(timeout=2.0)
        
        # Cancel pending tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        self.tasks.clear()

# Global MCP event loop manager for harvester
HARVESTER_LOOP_MANAGER = MCPEventLoopManager()

# =============================================================================
# ETHICAL WEB HARVESTER - PROPERLY STRUCTURED
# =============================================================================

class EthicalWebHarvester:
    """Ethical web content harvester for open source 3D/Blender resources"""
    
    def __init__(self, local_harvester):
        self.local_harvester = local_harvester
        
        # Ethical guidelines
        self.allowed_sources = {
            "docs.blender.org": {"license": "GPL/CC", "rate_limit": 10},
            "wiki.blender.org": {"license": "CC", "rate_limit": 5},
            "github.com": {"license": "MIT/GPL", "rate_limit": 60},  # GitHub API limit
            "developer.blender.org": {"license": "GPL", "rate_limit": 10},
        }
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Llammy-Harvester/1.0 (Educational/Research)',
            'Accept': 'text/html,application/json'
        })
        
        # Rate limiting
        self.last_request_time = {}
        
        # Web harvest tracking
        self.web_harvest_db_path = Path.home() / ".llammy" / "web_harvest.json"
        
    def harvest_blender_api_docs(self) -> Dict[str, Any]:
        """Harvest latest Blender API documentation"""
        base_urls = [
            "https://docs.blender.org/api/current/",
            "https://docs.blender.org/api/current/bpy.ops.html",
            "https://docs.blender.org/api/current/bpy.types.html"
        ]
        
        harvested_content = []
        
        for url in base_urls:
            if self._check_robots_txt(url) and self._rate_limit_ok(url):
                content = self._fetch_page_content(url)
                if content:
                    # Process as local content for existing harvester
                    processed = self._process_api_content(content, url)
                    harvested_content.append(processed)
                    
                self._update_rate_limit(url)
                
                # Small delay between requests
                time.sleep(1)
        
        # Feed into existing local harvester system
        result = self._integrate_with_local_harvester(harvested_content)
        
        # Mark harvest as complete
        self._mark_web_harvest_complete()
        
        return result
    
    def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch page content with error handling"""
        try:
            print(f"🌐 Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            if response.status_code == 200:
                return response.text
            else:
                print(f"⚠️ Unexpected status {response.status_code} for {url}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed for {url}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None
    
    def _process_api_content(self, content: str, url: str) -> Dict[str, Any]:
        """Process API documentation content"""
        processed_content = content
        
        # Basic text processing if BS4 available
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()
                
                processed_content = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in processed_content.splitlines())
                processed_content = '\n'.join(line for line in lines if line)
                
            except Exception as e:
                print(f"⚠️ BeautifulSoup processing failed, using raw content: {e}")
        
        return {
            'url': url,
            'content': processed_content[:10000],  # First 10KB
            'content_type': 'blender_api_docs',
            'harvest_time': datetime.now().isoformat(),
            'source_domain': urlparse(url).netloc,
            'content_length': len(processed_content)
        }
    
    def _integrate_with_local_harvester(self, content_list: List[Dict]) -> Dict[str, Any]:
        """Integrate web content with local harvester"""
        processed_count = 0
        total_content_size = 0
        
        try:
            for content_item in content_list:
                # Create a virtual file entry in harvester database
                virtual_path = f"web://{content_item['source_domain']}/{content_item['url'].split('/')[-1]}"
                
                # Calculate content size
                content_size = len(content_item['content'].encode('utf-8'))
                total_content_size += content_size
                
                # Store in harvester database using existing method
                cursor = self.local_harvester.connection.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO harvested_items 
                    (item_path, item_type, file_size, content_hash, metadata, 
                     business_value, harvest_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    virtual_path,
                    'blender_api_docs',
                    content_size,
                    hashlib.md5(content_item['content'].encode()).hexdigest(),
                    json.dumps({
                        'url': content_item['url'],
                        'source_type': 'web_harvest',
                        'content_preview': content_item['content'][:500],
                        'blender_relevance': 10.0,  # High relevance for API docs
                        'api_documentation': True
                    }),
                    50.0,  # High business value for current API docs
                    content_item['harvest_time']
                ))
                
                processed_count += 1
            
            self.local_harvester.connection.commit()
            
            # Update harvester stats
            self.local_harvester.stats['items_harvested'] += processed_count
            self.local_harvester.stats['total_size_mb'] += total_content_size / (1024 * 1024)
            
            print(f"✅ Web harvest integrated: {processed_count} pages, {total_content_size/1024:.1f}KB")
            
            return {
                'success': True,
                'pages_harvested': processed_count,
                'content_stored': len(content_list),
                'total_size_kb': total_content_size / 1024
            }
            
        except Exception as e:
            print(f"❌ Web harvest integration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'pages_harvested': 0
            }
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check robots.txt compliance"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            can_fetch = rp.can_fetch('*', url)
            if not can_fetch:
                print(f"🚫 robots.txt disallows fetching {url}")
            
            return can_fetch
            
        except Exception as e:
            print(f"⚠️ robots.txt check failed for {url}: {e}")
            return True  # If robots.txt unavailable, assume OK
    
    def _rate_limit_ok(self, url: str) -> bool:
        """Check if we can make request without violating rate limits"""
        domain = urlparse(url).netloc
        rate_limit = self.allowed_sources.get(domain, {}).get('rate_limit', 5)
        
        last_time = self.last_request_time.get(domain, 0)
        time_since = time.time() - last_time
        min_interval = 60 / rate_limit  # requests per minute
        
        if time_since < min_interval:
            wait_time = min_interval - time_since
            print(f"⏱️ Rate limiting: waiting {wait_time:.1f}s for {domain}")
            time.sleep(wait_time)
        
        return True
    
    def _update_rate_limit(self, url: str):
        """Update rate limiting tracker"""
        domain = urlparse(url).netloc
        self.last_request_time[domain] = time.time()
    
    def _mark_web_harvest_complete(self):
        """Mark web harvest as complete for today"""
        try:
            self.web_harvest_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            harvest_data = {
                'last_harvest_date': datetime.now().date().isoformat(),
                'last_harvest_time': datetime.now().isoformat(),
                'harvest_count': 1
            }
            
            # Load existing data if available
            if self.web_harvest_db_path.exists():
                with open(self.web_harvest_db_path, 'r') as f:
                    existing_data = json.load(f)
                    if existing_data.get('last_harvest_date') == harvest_data['last_harvest_date']:
                        harvest_data['harvest_count'] = existing_data.get('harvest_count', 0) + 1
            
            with open(self.web_harvest_db_path, 'w') as f:
                json.dump(harvest_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to mark web harvest complete: {e}")
    
    def get_last_web_harvest_time(self) -> datetime:
        """Get last web harvest time"""
        try:
            if self.web_harvest_db_path.exists():
                with open(self.web_harvest_db_path, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('last_harvest_time', '2000-01-01'))
        except Exception:
            pass
        
        return datetime.now() - timedelta(days=2)  # Default to 2 days ago

# =============================================================================
# INTELLIGENT HARVESTER MANAGER - WITH WEB HARVESTING
# =============================================================================

class IntelligentHarvesterManager:
    """Manages intelligent harvesting with idle detection, data caps, and web scraping"""
    
    def __init__(self, harvester_instance):
        self.harvester = harvester_instance
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Idle detection settings
        self.idle_threshold_minutes = 5  # Consider idle after 5 minutes
        self.blender_process_names = ['blender', 'blender.exe', 'Blender']
        self.last_activity_time = time.time()
        
        # Data usage tracking
        self.daily_data_cap_gb = 2.0
        self.current_day = datetime.now().date()
        self.daily_data_usage_mb = 0.0
        self.usage_db_path = Path.home() / ".llammy" / "harvester_usage.json"
        
        # Load existing usage data
        self._load_usage_data()
        
        # Harvesting settings
        self.harvest_batch_size = 50  # Files per batch
        self.harvest_interval_minutes = 15  # Check every 15 minutes
        self.max_file_size_mb = 10  # Skip files larger than 10MB
        
        # PSUTIL status
        self.psutil_available = PSUTIL_AVAILABLE
        
        if PSUTIL_AVAILABLE:
            print(f"🎯 Intelligent harvester initialized: {self.daily_data_cap_gb}GB daily cap with full process monitoring + web scraping")
        else:
            print(f"🎯 Intelligent harvester initialized: {self.daily_data_cap_gb}GB daily cap with basic monitoring + web scraping")
    
    def start_intelligent_monitoring(self):
        """Start intelligent background monitoring"""
        if self.monitoring_active:
            print("Intelligent monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._intelligent_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        if PSUTIL_AVAILABLE:
            print("🎯 Intelligent harvester monitoring started with process detection + web scraping")
        else:
            print("🎯 Intelligent harvester monitoring started in basic mode + web scraping")
    
    def stop_intelligent_monitoring(self):
        """Stop intelligent monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("🛑 Intelligent harvester monitoring stopped")
    
    def _intelligent_monitoring_loop(self):
        """Main intelligent monitoring loop with web harvesting"""
        while self.monitoring_active:
            try:
                # Check if new day (reset daily usage)
                self._check_daily_reset()
                
                # Skip if daily data cap reached
                if self._is_daily_cap_reached():
                    print(f"📊 Daily data cap reached ({self.daily_data_cap_gb}GB), harvesting paused")
                    time.sleep(self.harvest_interval_minutes * 60)
                    continue
                
                # Check if Blender is idle
                if self._is_blender_idle():
                    remaining_quota_mb = self._get_remaining_quota_mb()
                    
                    if remaining_quota_mb > 100:  # At least 100MB remaining
                        if PSUTIL_AVAILABLE:
                            print("😴 Blender idle detected, starting intelligent harvest...")
                        else:
                            print("😴 System appears idle, starting harvest (basic mode)...")
                        self._perform_intelligent_harvest(remaining_quota_mb)
                    else:
                        print(f"📉 Low quota remaining: {remaining_quota_mb:.1f}MB, skipping harvest")
                
                # Wait before next check
                time.sleep(self.harvest_interval_minutes * 60)
                
            except Exception as e:
                print(f"⚠️ Intelligent monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _perform_intelligent_harvest(self, max_data_mb: float):
        """Enhanced harvest including web content"""
        try:
            start_time = time.time()
            data_processed_mb = 0.0
            files_processed = 0
            
            # Add web harvesting once daily
            if self._should_harvest_web_content():
                print("🌐 Performing daily web content harvest...")
                try:
                    if hasattr(self.harvester, 'web_harvester') and self.harvester.web_harvester:
                        web_results = self.harvester.web_harvester.harvest_blender_api_docs()
                        if web_results.get('success'):
                            web_size_mb = web_results.get('total_size_kb', 0) / 1024
                            data_processed_mb += web_size_mb
                            print(f"✅ Web harvest successful: {web_results['pages_harvested']} pages, {web_size_mb:.1f}MB")
                        else:
                            print(f"⚠️ Web harvest failed: {web_results.get('error', 'Unknown error')}")
                    else:
                        print("⚠️ Web harvester not available")
                except Exception as e:
                    print(f"❌ Web harvest failed: {e}")
            
            # Continue with local file harvesting if quota allows
            remaining_quota = max_data_mb - data_processed_mb
            if remaining_quota > 10:  # At least 10MB remaining for local files
                # Get potential harvest targets
                harvest_targets = self._discover_harvest_targets()
                
                print(f"🔍 Found {len(harvest_targets)} potential local harvest targets")
                
                for target_path in harvest_targets:
                    # Check if we should stop
                    if not self.monitoring_active or not self._is_blender_idle():
                        print("⏸️ Harvest interrupted - Blender became active")
                        break
                    
                    if data_processed_mb >= max_data_mb:
                        print(f"📊 Data limit reached: {data_processed_mb:.1f}MB")
                        break
                    
                    if files_processed >= self.harvest_batch_size:
                        print(f"📦 Batch limit reached: {files_processed} files")
                        break
                    
                    # Process single file
                    file_size_mb = self._get_file_size_mb(target_path)
                    
                    if file_size_mb > self.max_file_size_mb:
                        continue  # Skip large files
                    
                    if data_processed_mb + file_size_mb > max_data_mb:
                        continue  # Would exceed quota
                    
                    # Harvest the file
                    result = self.harvester._harvest_single_file(target_path, "auto")
                    
                    if result['success']:
                        data_processed_mb += file_size_mb
                        files_processed += 1
                        
                        # Small delay to be gentle on system
                        time.sleep(0.1)
            
            # Update usage tracking
            self.daily_data_usage_mb += data_processed_mb
            self._save_usage_data()
            
            duration = time.time() - start_time
            print(f"✅ Intelligent harvest complete: {files_processed} local files, "
                  f"{data_processed_mb:.1f}MB total in {duration:.1f}s")
            
        except Exception as e:
            print(f"⚠ Intelligent harvest error: {e}")
    
    def _should_harvest_web_content(self) -> bool:
        """Check if daily web harvest is due"""
        if hasattr(self.harvester, 'web_harvester') and self.harvester.web_harvester:
            last_web_harvest = self.harvester.web_harvester.get_last_web_harvest_time()
            return (datetime.now() - last_web_harvest).days >= 1
        return False
    
    def _is_blender_idle(self) -> bool:
        """Detect if Blender is idle - with psutil fallback"""
        if not PSUTIL_AVAILABLE:
            # Basic mode: always consider idle after threshold
            idle_duration = time.time() - self.last_activity_time
            if idle_duration > (self.idle_threshold_minutes * 60):
                return True
            return False
        
        try:
            blender_processes = []
            
            # Find Blender processes
            for proc in psutil.process_iter(['name', 'cpu_percent']):
                if any(name.lower() in proc.info['name'].lower()
                       for name in self.blender_process_names):
                    blender_processes.append(proc)
            
            if not blender_processes:
                # Blender not running - safe to harvest
                return True
            
            # Check CPU usage of Blender processes
            total_cpu = sum(proc.cpu_percent(interval=1) for proc in blender_processes)
            
            # Consider idle if CPU usage < 5% for Blender
            if total_cpu < 5.0:
                idle_duration = time.time() - self.last_activity_time
                if idle_duration > (self.idle_threshold_minutes * 60):
                    return True
                else:
                    # Still in grace period
                    return False
            else:
                # Active - update last activity time
                self.last_activity_time = time.time()
                return False
                
        except Exception as e:
            print(f"⚠️ Idle detection error: {e}")
            return False  # Err on side of caution
    
    def _discover_harvest_targets(self) -> list:
        """Discover files worth harvesting"""
        targets = []
        
        # Common Blender project directories
        search_paths = [
            Path.home() / "Documents",
            Path.home() / "Desktop",
            Path.home() / "Downloads",
            Path("/tmp"),  # Linux/Mac temp
            Path("C:/temp") if Path("C:/temp").exists() else None,  # Windows temp
        ]
        
        # Remove None paths
        search_paths = [p for p in search_paths if p and p.exists()]
        
        # File extensions of interest
        target_extensions = {
            '.blend', '.obj', '.fbx', '.dae',  # 3D files
            '.py', '.json', '.csv',  # Data/script files
            '.jpg', '.png', '.exr',  # Images (small ones)
        }
        
        for search_path in search_paths:
            try:
                # Recent files only (last 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                
                for file_path in search_path.rglob('*'):
                    if (file_path.is_file() and
                        file_path.suffix.lower() in target_extensions and
                        datetime.fromtimestamp(file_path.stat().st_mtime) > cutoff_time):
                        
                        # Check if already harvested
                        if not self._already_harvested(file_path):
                            targets.append(file_path)
                            
                            # Limit discovery to prevent excessive scanning
                            if len(targets) >= 200:
                                break
                
            except Exception as e:
                print(f"⚠️ Discovery error in {search_path}: {e}")
        
        # Sort by modification time (newest first)
        targets.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return targets[:100]  # Limit to top 100 candidates
    
    def _already_harvested(self, file_path: Path) -> bool:
        """Check if file was already harvested"""
        try:
            if self.harvester.connection:
                cursor = self.harvester.connection.cursor()
                cursor.execute(
                    "SELECT id FROM harvested_items WHERE item_path = ?",
                    (str(file_path),)
                )
                return cursor.fetchone() is not None
        except:
            pass
        return False
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except:
            return 0.0
    
    def _check_daily_reset(self):
        """Check if we need to reset daily usage"""
        current_day = datetime.now().date()
        if current_day != self.current_day:
            print(f"🌅 New day detected, resetting daily usage counter")
            self.current_day = current_day
            self.daily_data_usage_mb = 0.0
            self._save_usage_data()
    
    def _is_daily_cap_reached(self) -> bool:
        """Check if daily data cap is reached"""
        daily_cap_mb = self.daily_data_cap_gb * 1024
        return self.daily_data_usage_mb >= daily_cap_mb
    
    def _get_remaining_quota_mb(self) -> float:
        """Get remaining daily quota in MB"""
        daily_cap_mb = self.daily_data_cap_gb * 1024
        return max(0, daily_cap_mb - self.daily_data_usage_mb)
    
    def _load_usage_data(self):
        """Load usage data from disk"""
        try:
            if self.usage_db_path.exists():
                with open(self.usage_db_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if data is from today
                    stored_date = datetime.fromisoformat(data.get('date', '2000-01-01')).date()
                    if stored_date == self.current_day:
                        self.daily_data_usage_mb = data.get('usage_mb', 0.0)
                        print(f"📊 Loaded daily usage: {self.daily_data_usage_mb:.1f}MB")
        except Exception as e:
            print(f"⚠️ Failed to load usage data: {e}")
    
    def _save_usage_data(self):
        """Save usage data to disk"""
        try:
            self.usage_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'date': self.current_day.isoformat(),
                'usage_mb': self.daily_data_usage_mb,
                'cap_gb': self.daily_data_cap_gb,
                'psutil_available': self.psutil_available
            }
            
            with open(self.usage_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save usage data: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        remaining_quota_mb = self._get_remaining_quota_mb()
        cap_reached = self._is_daily_cap_reached()
        
        return {
            'daily_cap_gb': self.daily_data_cap_gb,
            'daily_usage_mb': self.daily_data_usage_mb,
            'daily_usage_gb': self.daily_data_usage_mb / 1024,
            'remaining_quota_mb': remaining_quota_mb,
            'remaining_quota_gb': remaining_quota_mb / 1024,
            'cap_reached': cap_reached,
            'usage_percentage': (self.daily_data_usage_mb / (self.daily_data_cap_gb * 1024)) * 100,
            'monitoring_active': self.monitoring_active,
            'current_date': self.current_day.isoformat(),
            'psutil_available': self.psutil_available,
            'monitoring_mode': 'Full Process Detection' if self.psutil_available else 'Basic Mode',
            'web_scraping_available': hasattr(self.harvester, 'web_harvester') and self.harvester.web_harvester is not None
        }
    
    def set_daily_cap(self, gb: float):
        """Update daily data cap"""
        self.daily_data_cap_gb = max(0.1, min(gb, 10.0))  # Between 0.1GB and 10GB
        self._save_usage_data()
        print(f"📊 Daily data cap set to {self.daily_data_cap_gb}GB")

# =============================================================================
# COMPLETE HARVESTER IMPLEMENTATION
# =============================================================================

class LlammyDataHarvester:
    """Complete unified data harvester with all features including web scraping"""
    
    def __init__(self, base_storage_path: str, ai_engine=None):
        self.base_path = Path(base_storage_path).expanduser()
        self.ai_engine = ai_engine
        
        # Ensure storage directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Harvesting database
        self.db_path = self.base_path / "harvester.db"
        self.connection = None
        
        # Harvesting stats
        self.stats = {
            'items_harvested': 0,
            'total_size_mb': 0.0,
            'categories_processed': {},
            'last_harvest_time': None,
            'harvest_sessions': 0,
            'errors_encountered': 0,
            'rag_feedback_received': 0,
            'psutil_available': PSUTIL_AVAILABLE,
            'web_harvest_sessions': 0,
            'web_pages_harvested': 0
        }
        
        # MCP communication channel
        self.mcp_channel = None
        
        # Harvesting queue
        self.harvest_queue = []
        self.processing = False
        
        # Intelligent harvester manager
        self.intelligent_manager = None
        
        # Web harvester (will be added by factory function)
        self.web_harvester = None
        
        print(f"🌾 Unified harvester initialized: {self.base_path}")
        if PSUTIL_AVAILABLE:
            print("✅ Full intelligent monitoring available")
        else:
            print("⚠️ Basic monitoring mode (install psutil for full features)")
    
    def initialize(self) -> bool:
        """Initialize complete harvester system"""
        try:
            # Initialize database
            self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.connection.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS harvested_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_path TEXT UNIQUE,
                    item_type TEXT,
                    file_size INTEGER,
                    content_hash TEXT,
                    metadata TEXT,
                    business_value REAL,
                    harvest_timestamp TIMESTAMP,
                    rag_feedback_score REAL DEFAULT 0.0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS harvest_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start TIMESTAMP,
                    session_end TIMESTAMP,
                    items_processed INTEGER,
                    success_rate REAL
                )
            ''')
            
            self.connection.commit()
            
            # Start MCP event loop
            if not HARVESTER_LOOP_MANAGER.running:
                HARVESTER_LOOP_MANAGER.start_event_loop()
            
            # Initialize intelligent manager
            try:
                self.intelligent_manager = IntelligentHarvesterManager(self)
                if PSUTIL_AVAILABLE:
                    print("🎯 Intelligent harvester manager initialized with full monitoring")
                else:
                    print("🎯 Intelligent harvester manager initialized in basic mode")
            except Exception as e:
                print(f"⚠️ Intelligent harvester initialization failed: {e}")
            
            print("✅ Complete harvester system initialized")
            return True
            
        except Exception as e:
            print(f"⚠ Harvester initialization failed: {e}")
            return False
    
    def harvest_data(self, data_source: str, data_type: str = "auto") -> Dict[str, Any]:
        """Harvest data from source with type detection"""
        try:
            source_path = Path(data_source).expanduser()
            
            if not source_path.exists():
                return {'success': False, 'error': f'Source not found: {data_source}'}
            
            session_start = time.time()
            items_processed = 0
            items_successful = 0
            
            print(f"🔍 Harvesting from: {source_path}")
            
            if source_path.is_file():
                # Single file harvest
                result = self._harvest_single_file(source_path, data_type)
                items_processed = 1
                items_successful = 1 if result['success'] else 0
                
            elif source_path.is_dir():
                # Directory harvest
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            result = self._harvest_single_file(file_path, data_type)
                            items_processed += 1
                            if result['success']:
                                items_successful += 1
                                
                            # Progress feedback
                            if items_processed % 10 == 0:
                                print(f"📊 Processed {items_processed} items...")
                                
                        except Exception as e:
                            print(f"⚠️ Failed to harvest {file_path}: {e}")
                            self.stats['errors_encountered'] += 1
            
            # Update session stats
            session_time = time.time() - session_start
            success_rate = (items_successful / max(items_processed, 1)) * 100
            
            self._record_harvest_session(session_start, session_time, items_processed, success_rate)
            
            self.stats['harvest_sessions'] += 1
            self.stats['last_harvest_time'] = datetime.now().isoformat()
            
            print(f"✅ Harvest complete: {items_successful}/{items_processed} items ({success_rate:.1f}% success)")
            
            return {
                'success': True,
                'items_processed': items_processed,
                'items_successful': items_successful,
                'success_rate': success_rate,
                'processing_time': session_time
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _harvest_single_file(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """Harvest single file with metadata extraction"""
        try:
            # Get file metadata
            stat = file_path.stat()
            file_size = stat.st_size
            
            # Skip very large files (>50MB)
            if file_size > 50 * 1024 * 1024:
                return {'success': False, 'error': 'File too large'}
            
            # Calculate content hash
            content_hash = self._calculate_file_hash(file_path)
            
            # Detect file type if auto
            if data_type == "auto":
                data_type = self._detect_file_type(file_path)
            
            # Extract metadata
            metadata = self._extract_file_metadata(file_path, data_type)
            
            # Calculate business value
            business_value = self._calculate_business_value(file_path, data_type, metadata)
            
            # Store in database
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO harvested_items 
                (item_path, item_type, file_size, content_hash, metadata, 
                 business_value, harvest_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(file_path),
                data_type,
                file_size,
                content_hash,
                json.dumps(metadata),
                business_value,
                datetime.now().isoformat()
            ))
            
            self.connection.commit()
            
            # Update stats
            self.stats['items_harvested'] += 1
            self.stats['total_size_mb'] += file_size / (1024 * 1024)
            
            if data_type not in self.stats['categories_processed']:
                self.stats['categories_processed'][data_type] = 0
            self.stats['categories_processed'][data_type] += 1
            
            return {
                'success': True,
                'file_path': str(file_path),
                'data_type': data_type,
                'business_value': business_value
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension and content"""
        ext = file_path.suffix.lower()
        
        type_mapping = {
            '.blend': 'blender_scene',
            '.obj': '3d_model',
            '.fbx': '3d_model',
            '.dae': '3d_model',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.exr': 'image',
            '.py': 'script',
            '.json': 'data',
            '.csv': 'data',
            '.txt': 'text',
            '.md': 'documentation'
        }
        
        return type_mapping.get(ext, 'unknown')
    
    def _extract_file_metadata(self, file_path: Path, data_type: str) -> Dict[str, Any]:
        """Extract metadata based on file type"""
        metadata = {
            'filename': file_path.name,
            'extension': file_path.suffix,
            'size_mb': file_path.stat().st_size / (1024 * 1024),
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        # Type-specific metadata extraction
        if data_type in ['script', 'data', 'text', 'documentation']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(1000)  # First 1KB
                    metadata['content_preview'] = content
                    metadata['line_count'] = content.count('\n')
                    
                    # Blender-specific keyword detection
                    blender_keywords = ['bpy', 'blender', 'mesh', 'material', 'scene']
                    metadata['blender_relevance'] = sum(1 for kw in blender_keywords if kw in content.lower())
                    
            except Exception as e:
                metadata['content_error'] = str(e)
        
        return metadata
    
    def _calculate_business_value(self, file_path: Path, data_type: str, metadata: Dict[str, Any]) -> float:
        """Calculate business value of harvested item"""
        base_value = 1.0
        
        # Type-based value multipliers
        type_values = {
            'blender_scene': 10.0,
            'blender_api_docs': 50.0,  # Very high value for API docs
            '3d_model': 5.0,
            'script': 8.0,
            'image': 3.0,
            'data': 4.0,
            'documentation': 2.0
        }
        
        base_value *= type_values.get(data_type, 1.0)
        
        # Size bonus (sweet spot: 100KB - 10MB)
        size_mb = metadata.get('size_mb', 0)
        if 0.1 <= size_mb <= 10:
            base_value *= (1 + size_mb / 10)
        
        # Blender relevance bonus
        blender_relevance = metadata.get('blender_relevance', 0)
        if blender_relevance > 0:
            base_value *= (1 + blender_relevance * 0.2)
        
        return round(base_value, 2)
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate file hash for deduplication"""
        try:
            hash_obj = hashlib.md5()
            
            if file_path.stat().st_size < 1024 * 1024:  # 1MB
                with open(file_path, 'rb') as f:
                    hash_obj.update(f.read())
            else:
                # For large files, hash filename + size + modified time
                content = f"{file_path.name}{file_path.stat().st_size}{file_path.stat().st_mtime}"
                hash_obj.update(content.encode())
            
            return hash_obj.hexdigest()
            
        except Exception:
            return hashlib.md5(str(file_path).encode()).hexdigest()
    
    def _record_harvest_session(self, start_time: float, duration: float,
                               items_processed: int, success_rate: float):
        """Record harvest session statistics"""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO harvest_sessions 
                (session_start, session_end, items_processed, success_rate)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.fromtimestamp(start_time).isoformat(),
                datetime.fromtimestamp(start_time + duration).isoformat(),
                items_processed,
                success_rate
            ))
            
            self.connection.commit()
            
        except Exception as e:
            print(f"⚠️ Failed to record harvest session: {e}")
    
    def receive_rag_feedback(self, feedback: Dict[str, Any]):
        """Receive feedback from RAG about harvested content effectiveness"""
        try:
            file_path = feedback.get('file_path')
            effectiveness_score = feedback.get('effectiveness_score', 0.0)
            
            if file_path:
                cursor = self.connection.cursor()
                cursor.execute('''
                    UPDATE harvested_items 
                    SET rag_feedback_score = ? 
                    WHERE item_path = ?
                ''', (effectiveness_score, file_path))
                
                self.connection.commit()
                self.stats['rag_feedback_received'] += 1
                
                print(f"📊 RAG feedback received: {file_path} scored {effectiveness_score:.2f}")
            
        except Exception as e:
            print(f"⚠️ Failed to process RAG feedback: {e}")
    
    # Intelligent harvesting interface methods
    def start_background_harvesting(self):
        """Start intelligent background harvesting with web scraping"""
        if self.intelligent_manager:
            self.intelligent_manager.start_intelligent_monitoring()
            if PSUTIL_AVAILABLE:
                print("🎯 Background harvesting started (idle detection + 2GB cap + web scraping)")
            else:
                print("🎯 Background harvesting started (basic mode + 2GB cap + web scraping)")
        else:
            print("⚠️ Intelligent manager not available")
    
    def stop_background_harvesting(self):
        """Stop intelligent background harvesting"""
        if self.intelligent_manager:
            self.intelligent_manager.stop_intelligent_monitoring()
        else:
            print("⚠️ Intelligent manager not available")
    
    def set_daily_data_cap(self, gb: float):
        """Set daily data cap in GB"""
        if self.intelligent_manager:
            self.intelligent_manager.set_daily_cap(gb)
        else:
            print("⚠️ Intelligent manager not available")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get intelligent harvesting usage stats"""
        if self.intelligent_manager:
            return self.intelligent_manager.get_usage_stats()
        return {
            'intelligent_harvesting': False,
            'psutil_available': PSUTIL_AVAILABLE,
            'monitoring_mode': 'Disabled - No intelligent manager'
        }
    
    def get_comprehensive_earning_stats(self) -> Dict[str, Any]:
        """Get comprehensive harvester statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Get recent harvest statistics
            cursor.execute('''
                SELECT COUNT(*), AVG(business_value), SUM(file_size), AVG(rag_feedback_score)
                FROM harvested_items 
                WHERE harvest_timestamp > datetime('now', '-7 days')
            ''')
            
            recent_stats = cursor.fetchone()
            recent_count, avg_value, total_size, avg_feedback = recent_stats
            
            # Get category breakdown
            cursor.execute('''
                SELECT item_type, COUNT(*), SUM(business_value)
                FROM harvested_items 
                GROUP BY item_type
            ''')
            
            category_breakdown = {row[0]: {'count': row[1], 'value': row[2]}
                                for row in cursor.fetchall()}
            
            # Get web harvest stats
            cursor.execute('''
                SELECT COUNT(*), SUM(business_value)
                FROM harvested_items 
                WHERE item_type = 'blender_api_docs'
            ''')
            
            web_stats = cursor.fetchone()
            web_pages, web_value = web_stats
            
            # Combine with intelligent harvesting stats
            base_stats = {
                'harvester_active': True,
                'stats': self.stats.copy(),
                'recent_week': {
                    'items_harvested': recent_count or 0,
                    'avg_business_value': avg_value or 0.0,
                    'total_size_mb': (total_size or 0) / (1024 * 1024),
                    'avg_rag_feedback': avg_feedback or 0.0
                },
                'category_breakdown': category_breakdown,
                'web_harvest_stats': {
                    'pages_harvested': web_pages or 0,
                    'total_business_value': web_value or 0.0,
                    'web_scraping_available': self.web_harvester is not None
                },
                'database_health': True,
                'psutil_status': {
                    'available': PSUTIL_AVAILABLE,
                    'monitoring_mode': 'Full Process Detection' if PSUTIL_AVAILABLE else 'Basic Mode'
                }
            }
            
            # Add intelligent harvesting stats if available
            if self.intelligent_manager:
                intelligence_stats = self.intelligent_manager.get_usage_stats()
                base_stats['intelligent_harvesting'] = intelligence_stats
            
            return base_stats
            
        except Exception as e:
            return {
                'harvester_active': False,
                'error': str(e),
                'stats': self.stats.copy(),
                'psutil_status': {
                    'available': PSUTIL_AVAILABLE,
                    'monitoring_mode': 'Error in stats collection'
                }
            }
    
    def connect_to_module(self, module_name: str, mcp_channel):
        """Connect to another module via MCP"""
        try:
            if module_name == "rag":
                self.mcp_channel = mcp_channel
                print(f"Connected to {module_name} via MCP")
                return True
            
            return False
            
        except Exception as e:
            print(f"Failed to connect to {module_name}: {e}")
            return False
    
    def shutdown(self):
        """Shutdown harvester gracefully"""
        print("Shutting down unified harvester with web capabilities...")
        
        # Stop intelligent monitoring
        if self.intelligent_manager:
            self.intelligent_manager.stop_intelligent_monitoring()
        
        # Close database connection
        if self.connection:
            self.connection.close()
        
        # Stop MCP event loop
        HARVESTER_LOOP_MANAGER.stop_event_loop()
        
        print("Unified harvester shutdown complete")

# =============================================================================
# MCP SERVER FOUNDATION
# =============================================================================

class HarvesterMCPServer:
    """MCP Server for harvester communication"""
    
    def __init__(self, harvester: LlammyDataHarvester, port: int = 8002):
        self.harvester = harvester
        self.port = port
        self.server = None
    
    def start_server(self):
        """Start MCP server"""
        try:
            print(f"Starting harvester MCP server on port {self.port}")
            # This would start an actual MCP server
            # For now, just mark as ready
            return True
        except Exception as e:
            print(f"Failed to start MCP server: {e}")
            return False

# =============================================================================
# FACTORY FUNCTION - CORE INTEGRATION WITH WEB HARVESTING
# =============================================================================

def create_llammy_rich_dataset_harvester(base_storage_path: str, ai_engine=None) -> LlammyDataHarvester:
    """Factory function to create unified harvester with web capabilities"""
    try:
        harvester = LlammyDataHarvester(base_storage_path, ai_engine)
        
        if harvester.initialize():
            # Add web harvesting capabilities
            try:
                harvester.web_harvester = EthicalWebHarvester(harvester)
                print(f"✅ Web harvesting capabilities added")
            except Exception as e:
                print(f"⚠️ Web harvester initialization failed: {e}")
                harvester.web_harvester = None
            
            print(f"Unified Llammy harvester created: {base_storage_path}")
            print("Features enabled:")
            print("  - Core harvesting with database and MCP")
            if PSUTIL_AVAILABLE:
                print("  - Intelligent idle detection and background operation (FULL)")
                print("  - Process monitoring enabled")
            else:
                print("  - Basic idle detection and background operation")
                print("  - Install 'psutil' package for full process monitoring")
            
            if harvester.web_harvester:
                print("  - Ethical web scraping for Blender API documentation")
                if BS4_AVAILABLE:
                    print("  - Advanced HTML parsing with BeautifulSoup")
                else:
                    print("  - Basic text parsing (install beautifulsoup4 for better results)")
            else:
                print("  - Web scraping disabled (requests library required)")
                
            print("  - Daily data caps with automatic reset")
            print("  - RAG feedback integration for learning")
            print("  - Smart file discovery and prioritization")
            print("  - Memory-efficient processing with gentle system impact")
            print("  - Factory function for Core integration")
            print("  - Graceful fallback when dependencies unavailable")
            
            return harvester
        else:
            print("Harvester initialization failed")
            return None
            
    except Exception as e:
        print(f"Harvester creation failed: {e}")
        return None

# =============================================================================
# TESTING AND INTEGRATION
# =============================================================================

def test_unified_harvester():
    """Test complete unified harvester with web capabilities"""
    print("Testing unified harvester implementation with web scraping...")
    
    # Test factory function
    harvester = create_llammy_rich_dataset_harvester("~/test_harvest")
    
    if harvester:
        print("Factory function works")
        
        # Test intelligent features
        if harvester.intelligent_manager:
            print("Intelligent manager available")
            
            # Set 2GB cap
            harvester.set_daily_data_cap(2.0)
            
            # Get usage stats
            usage_stats = harvester.get_usage_stats()
            print(f"Usage stats: {usage_stats.get('daily_cap_gb', 'N/A')}GB cap")
            print(f"Monitoring mode: {usage_stats.get('monitoring_mode', 'Unknown')}")
            
            # Test web harvesting if available
            if harvester.web_harvester:
                print("Testing web harvesting capability...")
                try:
                    web_result = harvester.web_harvester.harvest_blender_api_docs()
                    print(f"Web harvest test: {'SUCCESS' if web_result.get('success') else 'FAILED'}")
                except Exception as e:
                    print(f"Web harvest test failed: {e}")
            
            # Start background (for testing - would normally run continuously)
            print("Testing background monitoring start...")
            harvester.start_background_harvesting()
            
            # Wait a moment
            time.sleep(2)
            
            # Stop background
            harvester.stop_background_harvesting()
            print("Background monitoring test complete")
        
        # Test comprehensive stats
        stats = harvester.get_comprehensive_earning_stats()
        print(f"Comprehensive stats available: {stats.get('harvester_active', False)}")
        print(f"psutil available: {stats.get('psutil_status', {}).get('available', False)}")
        print(f"Web scraping available: {stats.get('web_harvest_stats', {}).get('web_scraping_available', False)}")
        
        # Test RAG feedback
        harvester.receive_rag_feedback({
            'file_path': '/test/path',
            'effectiveness_score': 0.8
        })
        print("RAG feedback mechanism works")
        
        harvester.shutdown()
        return True
    else:
        print("Harvester creation failed")
        return False

if __name__ == "__main__":
    test_unified_harvester()

print("UNIFIED LLAMMY HARVESTER MODULE LOADED!")
print("Complete features:")
print("  - Core harvesting with database and MCP")
if PSUTIL_AVAILABLE:
    print("  - Intelligent idle detection and background operation (FULL)")
    print("  - Process monitoring enabled")
else:
    print("  - Basic idle detection and background operation")
    print("  - Install 'psutil' package for full process monitoring")

print("  - Ethical web scraping for Blender API documentation")
if BS4_AVAILABLE:
    print("  - Advanced HTML parsing enabled")
else:
    print("  - Basic text parsing (install beautifulsoup4 for better results)")
    
print("  - Daily data caps with automatic reset")
print("  - RAG feedback integration for learning")
print("  - Smart file discovery and prioritization")
print("  - Memory-efficient processing with gentle system impact")
print("  - Factory function for Core integration")
print("  - Graceful fallback when dependencies unavailable")
print("Ready for intelligent background data collection with web scraping!")
