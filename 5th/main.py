
"""
Main demonstration script for polygon monotonicity checking and triangulation.
"""

from core.monotonicitytester import main as monotonicity_demo, Direction, Polygon, Point, MonotonicityTester
from core.triangulation import demonstrate_triangulation, ChainType, MonotoneTriangulator, Triangle, TriangulationVisualizer

if __name__ == "__main__":
	print("\n" + "="*80)
	print("POLYGON MONOTONICITY CHECKING DEMONSTRATION")
	print("="*80 + "\n")
	monotonicity_demo()

	print("\n" + "="*80)
	print("MONOTONE POLYGON TRIANGULATION DEMONSTRATION")
	print("="*80 + "\n")
	demonstrate_triangulation()
