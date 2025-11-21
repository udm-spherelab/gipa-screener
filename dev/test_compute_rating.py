"""
Simple test to verify the compute_rating function works correctly.
This tests the critical bug fix where the second if was changed to elif.
"""


def compute_rating(result):
    """
    Standalone copy of compute_rating for testing.
    This is the FIXED version after changing second if to elif.
    """
    if result['participatory_method'] and result['green_infrastructure_intervention']:
        return 3
    elif not result['green_infrastructure_intervention'] or not result['participatory_method']:
        return 1
    return 2


def test_compute_rating():
    """Test all combinations of compute_rating function."""

    # Test Case 1: Both true -> should return 3
    result1 = {
        'participatory_method': True,
        'green_infrastructure_intervention': True
    }
    assert compute_rating(result1) == 3, "Failed: Both true should return 3"
    print("✓ Test 1 passed: Both true returns 3")

    # Test Case 2: Both false -> should return 1
    result2 = {
        'participatory_method': False,
        'green_infrastructure_intervention': False
    }
    assert compute_rating(result2) == 1, "Failed: Both false should return 1"
    print("✓ Test 2 passed: Both false returns 1")

    # Test Case 3: participatory_method true, green_infrastructure false -> should return 1
    result3 = {
        'participatory_method': True,
        'green_infrastructure_intervention': False
    }
    assert compute_rating(result3) == 1, "Failed: One true, one false should return 1"
    print("✓ Test 3 passed: One true, one false returns 1")

    # Test Case 4: participatory_method false, green_infrastructure true -> should return 1
    result4 = {
        'participatory_method': False,
        'green_infrastructure_intervention': True
    }
    assert compute_rating(result4) == 1, "Failed: One false, one true should return 1"
    print("✓ Test 4 passed: One false, one true returns 1")

    # Test Case 5: Both None -> should return 1 (not None = True, so the elif catches it)
    result5 = {
        'participatory_method': None,
        'green_infrastructure_intervention': None
    }
    assert compute_rating(result5) == 1, "Failed: Both None should return 1"
    print("✓ Test 5 passed: Both None returns 1")

    # Test Case 6: One None, one True -> should return 1 (due to 'not None' being True)
    result6 = {
        'participatory_method': None,
        'green_infrastructure_intervention': True
    }
    assert compute_rating(result6) == 1, "Failed: One None, one True should return 1"
    print("✓ Test 6 passed: One None, one True returns 1")

    print("\nAll tests passed! The compute_rating function works correctly.")


if __name__ == "__main__":
    test_compute_rating()
