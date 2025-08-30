# tests/test_no_internet.py
# Test to ensure no network modules are imported (hackathon compliance)

import sys
import importlib
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_no_network_imports():
    """
    Test that no banned network modules are imported.
    This ensures compliance with hackathon rules: no internet calls.
    """
    print("Testing for banned network module imports...")
    
    # List of banned network-related modules
    banned_modules = [
        'requests',
        'urllib', 'urllib2', 'urllib3', 
        'http.client', 'httplib',
        'socket',  # Note: some ML libraries might use socket internally
        'websocket', 'websockets',
        'aiohttp', 'httpx',
        'ftplib', 'smtplib',
        'telnetlib', 'poplib', 'imaplib',
        'nntplib'
    ]
    
    # Check currently imported modules
    imported_banned = []
    for module_name in banned_modules:
        if module_name in sys.modules:
            imported_banned.append(module_name)
    
    if imported_banned:
        print(f"❌ BANNED NETWORK MODULES DETECTED: {imported_banned}")
        raise AssertionError(
            f"Network modules imported: {imported_banned} - "
            "Remove all network calls to comply with hackathon rules"
        )
    
    print("✓ No banned network modules detected in current imports")

def test_strategy_imports():
    """
    Test that importing strategies doesn't bring in network modules.
    """
    print("Testing strategy imports for network dependencies...")
    
    # Record modules before importing strategies
    modules_before = set(sys.modules.keys())
    
    try:
        # Import strategies
        from strategies.strategy_vasudevjamdagnigaur_baseline import Strategy as BaselineStrategy
        from strategies.strategy_vasudevjamdagnigaur_ml import Strategy as MLStrategy
        
        # Record modules after importing
        modules_after = set(sys.modules.keys())
        new_modules = modules_after - modules_before
        
        print(f"New modules imported: {len(new_modules)}")
        
        # Check if any new modules are network-related
        banned_modules = {
            'requests', 'urllib', 'urllib2', 'urllib3', 
            'http.client', 'httplib', 'websocket', 'websockets',
            'aiohttp', 'httpx', 'ftplib', 'smtplib'
        }
        
        network_modules = new_modules & banned_modules
        if network_modules:
            raise AssertionError(f"Strategy imports brought in network modules: {network_modules}")
        
        print("✓ Strategy imports are clean of network dependencies")
        
    except ImportError as e:
        print(f"⚠ Warning: Could not import strategies: {e}")

def test_utils_compliance():
    """
    Test that utils module has network checking functionality.
    """
    print("Testing utils module network compliance...")
    
    try:
        from utils import assert_no_network_imports
        
        # This should pass if no network modules are imported
        assert_no_network_imports()
        print("✓ Utils network check passed")
        
    except ImportError:
        print("⚠ Warning: Could not import utils.assert_no_network_imports")
    except Exception as e:
        print(f"❌ Network check failed: {e}")
        raise

def simulate_network_import_detection():
    """
    Simulate importing a network module to test detection.
    """
    print("Testing network import detection...")
    
    # Temporarily add a fake network module to sys.modules
    fake_module_name = 'requests'
    if fake_module_name not in sys.modules:
        # Create a fake module object
        import types
        fake_module = types.ModuleType(fake_module_name)
        sys.modules[fake_module_name] = fake_module
        
        try:
            from utils import assert_no_network_imports
            
            # This should now fail
            try:
                assert_no_network_imports()
                print("❌ Network detection failed - should have caught fake requests module")
                raise AssertionError("Network detection is not working properly")
            except RuntimeError as e:
                print(f"✓ Network detection working: {e}")
            
        finally:
            # Clean up
            del sys.modules[fake_module_name]
    else:
        print("⚠ Skipping simulation - requests already imported")

def run_network_tests():
    """Run all network compliance tests."""
    print("=" * 60)
    print("HACKATHON NETWORK COMPLIANCE TESTS")
    print("=" * 60)
    
    try:
        test_no_network_imports()
        test_strategy_imports()
        test_utils_compliance()
        simulate_network_import_detection()
        
        print("\n" + "=" * 60)
        print("✅ ALL NETWORK TESTS PASSED - No internet dependencies detected!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ NETWORK TEST FAILED: {e}")
        print("Fix network dependencies before submission!")
        raise

if __name__ == "__main__":
    run_network_tests()

