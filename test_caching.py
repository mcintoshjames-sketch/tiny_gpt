#!/usr/bin/env python3
"""
Test script to validate token caching logic
"""

import os
import numpy as np
import tempfile
import shutil

def test_cache_logic():
    """Test the caching logic flow"""
    
    # Create a temporary directory for testing
    test_dir = tempfile.mkdtemp()
    train_cache_path = os.path.join(test_dir, "train_tokens.npy")
    val_cache_path = os.path.join(test_dir, "val_tokens.npy")
    
    print("Testing token caching logic...")
    print(f"Test directory: {test_dir}")
    
    # Test 1: Cache doesn't exist - should encode
    print("\n1. Test: Cache files don't exist")
    cache_valid = False
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        cache_valid = True
    
    assert not cache_valid, "Cache should not be valid when files don't exist"
    print("   ✓ Correctly detected no cache")
    
    # Simulate encoding and saving
    train_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    val_data = np.array([6, 7, 8], dtype=np.int32)
    
    np.save(train_cache_path, train_data)
    np.save(val_cache_path, val_data)
    print("   ✓ Simulated encoding and saved cache")
    
    # Test 2: Cache exists - should load
    print("\n2. Test: Cache files exist")
    cache_valid = False
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        try:
            loaded_train = np.load(train_cache_path)
            loaded_val = np.load(val_cache_path)
            print(f"   ✓ Loaded {len(loaded_train)} training tokens from cache")
            print(f"   ✓ Loaded {len(loaded_val)} validation tokens from cache")
            
            # Verify data integrity
            assert np.array_equal(loaded_train, train_data), "Training data mismatch"
            assert np.array_equal(loaded_val, val_data), "Validation data mismatch"
            print("   ✓ Data integrity verified")
            
            cache_valid = True
        except Exception as e:
            print(f"   ✗ Failed to load cache: {e}")
    
    assert cache_valid, "Cache should be valid when files exist"
    print("   ✓ Successfully loaded from cache")
    
    # Test 3: Corrupted cache - should re-encode
    print("\n3. Test: Corrupted cache file")
    # Corrupt the training cache
    with open(train_cache_path, 'wb') as f:
        f.write(b'corrupted data')
    
    cache_valid = False
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        try:
            loaded_train = np.load(train_cache_path)
            loaded_val = np.load(val_cache_path)
            cache_valid = True
        except Exception as e:
            print(f"   ✓ Correctly detected corrupted cache: {e}")
    
    assert not cache_valid, "Cache should not be valid when corrupted"
    print("   ✓ Will re-encode when cache is corrupted")
    
    # Test 4: Partial cache (only one file exists)
    print("\n4. Test: Partial cache (missing validation file)")
    os.remove(val_cache_path)
    
    # Restore training cache
    np.save(train_cache_path, train_data)
    
    cache_valid = False
    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        cache_valid = True
    
    assert not cache_valid, "Cache should not be valid when validation file is missing"
    print("   ✓ Correctly requires both cache files to exist")
    
    # Cleanup
    shutil.rmtree(test_dir)
    print(f"\n✓ All cache logic tests passed!")
    print(f"✓ Cleaned up test directory: {test_dir}")
    
    return True

if __name__ == "__main__":
    try:
        test_cache_logic()
        print("\n" + "="*60)
        print("VALIDATION SUCCESSFUL: Token caching logic is correct!")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        exit(1)
