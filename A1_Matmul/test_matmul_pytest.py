#!/usr/bin/env python3
"""
Matrix Multiplication Test Suite with Scoring
Usage: python test_matmul.py [--verbose]
"""

import pytest
import sys
import argparse
from matmul import matrix_multiply


# Test scoring weights
TEST_SCORES = {
    'test_basic_2x2': 15,
    'test_different_dimensions': 15, 
    'test_1x1_matrix': 10,
    'test_larger_matrix': 15,
    'test_zero_matrix': 10,
    'test_incompatible_dimensions': 10,
    'test_empty_matrix': 10,
    'test_non_numeric': 15
}

TOTAL_POINTS = sum(TEST_SCORES.values())


class TestMatrixMultiplication:
    
    def test_basic_2x2(self):
        """Test basic 2x2 matrix multiplication"""
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[5, 6], [7, 8]]
        expected = [[19, 22], [43, 50]]
        result = matrix_multiply(matrix1, matrix2)
        assert result == expected
    
    def test_different_dimensions(self):
        """Test multiplication with different compatible dimensions"""
        matrix1 = [[1, 2, 3], [4, 5, 6]]
        matrix2 = [[1, 2], [3, 4], [5, 6]]
        expected = [[22, 28], [49, 64]]
        result = matrix_multiply(matrix1, matrix2)
        assert result == expected
    
    def test_1x1_matrix(self):
        """Test 1x1 matrix multiplication"""
        matrix1 = [[5]]
        matrix2 = [[10]]
        expected = [[50]]
        result = matrix_multiply(matrix1, matrix2)
        assert result == expected
    
    def test_larger_matrix(self):
        """Test larger matrix multiplication with identity"""
        matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        matrix2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = matrix_multiply(matrix1, matrix2)
        assert result == expected
    
    def test_zero_matrix(self):
        """Test multiplication with zero matrix"""
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[0, 0], [0, 0]]
        expected = [[0, 0], [0, 0]]
        result = matrix_multiply(matrix1, matrix2)
        assert result == expected
    
    def test_incompatible_dimensions(self):
        """Test error handling for incompatible dimensions"""
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        with pytest.raises(ValueError):
            matrix_multiply(matrix1, matrix2)
    
    def test_empty_matrix(self):
        """Test error handling for empty matrices"""
        matrix1 = []
        matrix2 = [[1, 2], [3, 4]]
        with pytest.raises(ValueError):
            matrix_multiply(matrix1, matrix2)
    
    def test_non_numeric(self):
        """Test error handling for non-numeric elements"""
        matrix1 = [[1, "a"], [3, 4]]
        matrix2 = [[1, 2], [3, 4]]
        with pytest.raises(TypeError):
            matrix_multiply(matrix1, matrix2)


def calculate_score(test_results):
    """Calculate score based on test results"""
    earned_points = 0
    for test_name, outcome in test_results.items():
        if outcome == 'PASSED' and test_name in TEST_SCORES:
            earned_points += TEST_SCORES[test_name]
    
    return int((earned_points / TOTAL_POINTS) * 100)


def main():
    parser = argparse.ArgumentParser(description="Matrix multiplication test suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show detailed test results")
    args = parser.parse_args()
    
    if args.verbose:
        # Run pytest with verbose output
        exit_code = pytest.main([__file__, "-v"])
    else:
        # Run pytest programmatically and collect results
        class TestCollector:
            def __init__(self):
                self.results = {}
            
            def pytest_runtest_logreport(self, report):
                if report.when == 'call':
                    test_name = report.nodeid.split('::')[-1]
                    self.results[test_name] = 'PASSED' if report.outcome == 'passed' else 'FAILED'
        
        collector = TestCollector()
        exit_code = pytest.main([__file__, "-q", "--tb=no"], plugins=[collector])
        
        score = calculate_score(collector.results)
        print(score)


if __name__ == "__main__":
    main()
