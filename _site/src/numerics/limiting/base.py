# ------------------------------------------------------------------------ #
#
#       quail: A lightweight discontinuous Galerkin code for
#              teaching and prototyping
#		<https://github.com/IhmeGroup/quail>
#       
#		Copyright (C) 2020-2021
#
#       This program is distributed under the terms of the GNU
#		General Public License v3.0. You should have received a copy
#       of the GNU General Public License along with this program.  
#		If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------ #

# ------------------------------------------------------------------------ #
#
#       File : src/numerics/limiting/base.py
#
#       Contains class definition for the limiter abstract base class.
#
# ------------------------------------------------------------------------ #
from abc import ABC, abstractmethod
import numpy as np

import errors
import general


class LimiterBase(ABC):
	'''
	This is a base class for any limiter type.

	Abstract Constants:
	-------------------
	COMPATIBLE_PHYSICS_TYPES
	    physics types compatible with the limiter; either
	    general.PhysicsType enum member or iterable of enum members

	Methods:
	--------
	check_compatibility
		checks for compatibility between physics and limiter
	precompute_helpers
		precomputes helper arrays
	limit_solution
		applies limiter to global solution
	'''
	@property
	@abstractmethod
	def COMPATIBLE_PHYSICS_TYPES(self):
		'''
		physics types compatible with the limiter; either
	    general.PhysicsType enum member or iterable of enum members
	    '''
		pass

	def __init__(self, physics_type):
		self.check_compatibility(physics_type)

	def check_compatibility(self, physics_type):
		'''
		This method checks for compatibility with the given physics type.

		Inputs:
		-------
			physics_type: physics type (general.PhysicsType enum member)
		'''
		try:
			if physics_type not in self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError
		except TypeError:
			if physics_type != self.COMPATIBLE_PHYSICS_TYPES:
				raise errors.IncompatibleError

	@abstractmethod
	def precompute_helpers(self, solver):
		'''
		This method precomputes helper arrays

		Inputs:
		-------
			solver: solver object
		'''
		pass

	@abstractmethod
	def limit_solution(self, solver, Uc):
		'''
		This method limits the global solution

		Inputs:
		-------
			solver: solver object
			Uc: state coefficients of global solution
				[num_elems, nb, ns]

		Outputs:
		--------
			Uc: state coefficients of global solution
				[num_elems, nb, ns] (modified)
		'''
		pass
