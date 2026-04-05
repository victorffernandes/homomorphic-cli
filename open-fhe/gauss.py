"""
Gaussian Elimination Solver with Partial Pivoting
Finds exact solutions (or detects no solution / infinite solutions) for general linear systems.
"""

import copy
from typing import List, Tuple, Optional


class GaussianSolver:
    """
    Solves Ax = b using Gaussian elimination with partial pivoting.
    Handles rank-deficient systems and provides detailed solution classification.
    """

    def __init__(self, A: List[List[float]], b: List[float], tolerance: float = 1e-10):
        """
        Initialize solver with system Ax = b.
        
        Args:
            A: m x n coefficient matrix (list of rows)
            b: m x 1 right-hand side vector
            tolerance: threshold for treating values as zero (for numerical stability)
        """
        self.m = len(A)  # number of equations
        self.n = len(A[0]) if A else 0  # number of variables
        self.tolerance = tolerance
        
        # Augmented matrix [A | b]
        self.augmented = [A[i] + [b[i]] for i in range(self.m)]
        
        # Track row operations for diagnostics
        self.pivot_rows = []
        self.pivot_cols = []
        self.rank = 0
        
    def _abs_max_in_column(self, col: int, start_row: int) -> int:
        """Find row with maximum absolute value in column col, starting from start_row."""
        max_row = start_row
        max_val = abs(self.augmented[start_row][col])
        
        for row in range(start_row + 1, self.m):
            val = abs(self.augmented[row][col])
            if val > max_val:
                max_val = val
                max_row = row
        
        return max_row
    
    def _swap_rows(self, row1: int, row2: int):
        """Swap two rows in augmented matrix."""
        self.augmented[row1], self.augmented[row2] = self.augmented[row2], self.augmented[row1]
    
    def _eliminate_below(self, pivot_row: int, pivot_col: int):
        """Eliminate entries below pivot using row operations."""
        pivot_val = self.augmented[pivot_row][pivot_col]
        
        for row in range(pivot_row + 1, self.m):
            if abs(self.augmented[row][pivot_col]) < self.tolerance:
                continue
            
            factor = self.augmented[row][pivot_col] / pivot_val
            for col in range(pivot_col, self.n + 1):
                self.augmented[row][col] -= factor * self.augmented[pivot_row][col]
    
    def forward_elimination(self) -> bool:
        """
        Perform forward elimination with partial pivoting.
        Returns True if system is consistent (solution may exist).
        """
        current_row = 0
        
        for col in range(self.n):
            # Find pivot in column col from current_row onwards
            pivot_row = self._abs_max_in_column(col, current_row)
            
            # If pivot is negligible, this column is dependent
            if abs(self.augmented[pivot_row][col]) < self.tolerance:
                continue
            
            # Swap rows to bring pivot to current_row
            if pivot_row != current_row:
                self._swap_rows(current_row, pivot_row)
            
            # Record pivot position
            self.pivot_rows.append(current_row)
            self.pivot_cols.append(col)
            self.rank += 1
            
            # Eliminate below
            self._eliminate_below(current_row, col)
            current_row += 1
        
        # Check for inconsistency: row of zeros with non-zero RHS
        for row in range(self.rank, self.m):
            if all(abs(self.augmented[row][col]) < self.tolerance for col in range(self.n)):
                if abs(self.augmented[row][self.n]) > self.tolerance:
                    return False
        
        return True
    
    def back_substitution(self) -> Optional[List[float]]:
        """
        Perform back substitution on upper triangular form.
        Returns solution vector if unique solution exists, None otherwise.
        """
        if self.rank < self.n:
            return None
        
        x = [0.0] * self.n
        
        for i in range(len(self.pivot_rows) - 1, -1, -1):
            row_idx = self.pivot_rows[i]
            col_idx = self.pivot_cols[i]
            
            rhs = self.augmented[row_idx][self.n]
            for j in range(col_idx + 1, self.n):
                rhs -= self.augmented[row_idx][j] * x[j]
            
            pivot_val = self.augmented[row_idx][col_idx]
            if abs(pivot_val) < self.tolerance:
                return None
            
            x[col_idx] = rhs / pivot_val
        
        return x
    
    def solve(self) -> Tuple[Optional[List[float]], str]:
        """
        Solve the system and return (solution, status).
        Status: "unique", "infinite", or "no_solution"
        """
        is_consistent = self.forward_elimination()
        
        if not is_consistent:
            return None, "no_solution"
        
        if self.rank < self.n:
            return None, "infinite"
        
        x = self.back_substitution()
        if x is None:
            return None, "infinite"
        
        return x, "unique"
    
    def residual(self, x: List[float], A_original: List[List[float]], b_original: List[float]) -> float:
        """Compute Euclidean norm of residual ||b - Ax||."""
        residual_vec = []
        for i in range(len(b_original)):
            r = b_original[i] - sum(A_original[i][j] * x[j] for j in range(len(x)))
            residual_vec.append(r)
        
        return sum(r**2 for r in residual_vec) ** 0.5
    
    def print_solution_details(self, x: Optional[List[float]], status: str,
                               A_original: List[List[float]], b_original: List[float]):
        """Print detailed solution information."""
        print("\n" + "=" * 60)
        print(f"System dimensions: {self.m} equations, {self.n} variables")
        print(f"Rank: {self.rank}")
        print(f"Solution status: {status}")
        print("=" * 60)
        
        if status == "unique":
            print(f"\nSolution x = {[round(v, 6) for v in x]}")
            res = self.residual(x, A_original, b_original)
            print(f"Residual ||b - Ax|| = {res:.2e}")
            
            Ax = [sum(A_original[i][j] * x[j] for j in range(len(x))) for i in range(len(b_original))]
            print(f"Ax = {[round(v, 6) for v in Ax]}")
            print(f"b  = {[round(v, 6) for v in b_original]}")
        
        elif status == "infinite":
            print(f"\nInfinite solutions exist.")
            print(f"Underdetermined (rank {self.rank} < variables {self.n}).")
            print(f"Degrees of freedom: {self.n - self.rank}")
        
        elif status == "no_solution":
            print(f"\nNo solution exists.")
            print(f"System is inconsistent.")


def solve_system(A: List[List[float]], b: List[float], verbose: bool = True) -> Tuple[Optional[List[float]], str]:
    """
    Convenience function to solve Ax = b.
    
    Returns: (solution, status) where status is "unique", "infinite", or "no_solution"
    """
    solver = GaussianSolver(A, b)
    A_original = copy.deepcopy(A)
    b_original = copy.deepcopy(b)
    
    x, status = solver.solve()
    
    if verbose:
        solver.print_solution_details(x, status, A_original, b_original)
    
    return x, status


if __name__ == "__main__":
    # Example 1: Unique solution (diagonally dominant)
    print("\n" + "#" * 60)
    print("EXAMPLE 1: Unique Solution (Diagonally Dominant)")
    print("#" * 60)
    A1 = [
        [4.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 4.0],
    ]
    b1 = [15.0, 10.0, 15.0]
    x1, status1 = solve_system(A1, b1)
    
    # Example 2: Unique solution (requires pivoting)
    print("\n" + "#" * 60)
    print("EXAMPLE 2: Unique Solution (Pivoting Needed)")
    print("#" * 60)
    A2 = [
        [0.0, 2.0, 3.0],
        [1.0, -1.0, 1.0],
        [2.0, 1.0, 1.0],
    ]
    b2 = [8.0, 2.0, 5.0]
    x2, status2 = solve_system(A2, b2)
    
    # Example 3: Infinite solutions
    print("\n" + "#" * 60)
    print("EXAMPLE 3: Infinite Solutions (Underdetermined)")
    print("#" * 60)
    A3 = [
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    ]
    b3 = [5.0, 10.0]
    x3, status3 = solve_system(A3, b3)
    
    # Example 4: No solution (inconsistent)
    print("\n" + "#" * 60)
    print("EXAMPLE 4: No Solution (Inconsistent)")
    print("#" * 60)
    A4 = [
        [1.0, 2.0],
        [2.0, 4.0],
        [1.0, 2.0],
    ]
    b4 = [3.0, 6.0, 5.0]
    x4, status4 = solve_system(A4, b4)
