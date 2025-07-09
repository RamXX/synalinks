# Modified from: keras/src/ops/optimizer.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import warnings

import docstring_parser

from synalinks.src import backend
from synalinks.src.backend import DataModel
from synalinks.src.backend import contains_schema
from synalinks.src.backend import standardize_schema
from synalinks.src.initializers import Empty
from synalinks.src.metrics import Metric
from synalinks.src.modules import Module
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils import tracking
from synalinks.src.utils.naming import auto_name


class Iteration(DataModel):
    iteration: int = 0


class Optimizer(SynalinksSaveable):
    """Optimizer base class: all Synalinks optimizers inherit from this class.

    Args:
        schema (dict): The schema of the variables that the optimizer can act upon.
        data_model (DataModel): The backend data model that the optimizer can act upon,
            if no schema is specified, uses the data model to infer it.
        name (str): The name of the optimizer.
        description (str): The description of the optimizer.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        name=None,
        description=None,
        **kwargs,
    ):
        self._lock = False

        if kwargs:
            raise ValueError(f"Argument(s) not recognized: {kwargs}")

        if name is None:
            name = auto_name(self.__class__.__name__)
        self.name = name

        if description is None:
            if self.__class__.__doc__:
                description = docstring_parser.parse(
                    self.__class__.__doc__
                ).short_description
            else:
                description = ""
        self.description = description

        if not data_model and not schema:
            raise ValueError(
                "You should provide at least one argument "
                "between `data_model` or `schema`"
            )
        if not schema and data_model:
            schema = standardize_schema(data_model.get_schema())
        self._schema = schema

        self.built = False

        self._initialize_tracker()

        with backend.name_scope(self.name, caller=self):
            iterations = backend.Variable(
                initializer=Empty(data_model=Iteration),
                data_model=Iteration,
                trainable=False,
                name="iteration",
            )
        # self._track_variable(iterations)
        self._iteration = iterations

    @tracking.no_automatic_dependency_tracking
    def _initialize_tracker(self):
        if hasattr(self, "_tracker"):
            return

        trainable_variables = []
        non_trainable_variables = []
        modules = []
        self._tracker = tracking.Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and not x.trainable,
                    non_trainable_variables,
                ),
                "modules": (
                    lambda x: isinstance(x, Module) and not isinstance(x, Metric),
                    modules,
                ),
            },
            exclusions={"non_trainable_variables": ["trainable_variables"]},
        )
        self._trainable_variables = trainable_variables
        self._non_trainable_variables = non_trainable_variables
        self._modules = modules

    def __setattr__(self, name, value):
        # Track Variables, Modules, Metrics.
        if name != "_tracker":
            if not hasattr(self, "_tracker"):
                self._initialize_tracker()
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def get_schema(self):
        return self._schema

    @property
    def variables(self):
        return self._non_trainable_variables[:] + self._trainable_variables[:]

    @property
    def non_trainable_variables(self):
        return self._non_trainable_variables[:]

    @property
    def trainable_variables(self):
        variables = []
        for module in self._modules:
            variables.extend(module.trainable_variables)
        return variables

    @property
    def iterations(self):
        return self._iteration

    def save_own_variables(self, store):
        """Get the state of this optimizer object."""
        for i, variable in enumerate(self.variables):
            store[str(i)] = variable.numpy()

    def load_own_variables(self, store):
        """Set the state of this optimizer object."""
        if len(store.keys()) != len(self.variables):
            msg = (
                f"Skipping variable loading for optimizer '{self.name}', "
                f"because it has {len(self.variables)} variables whereas "
                f"the saved optimizer has {len(store.keys())} variables. "
            )
            if len(self.variables) == 0:
                msg += (
                    "This is likely because the optimizer has not been called/built yet."
                )
            warnings.warn(msg, stacklevel=2)
            return
        for i, variable in enumerate(self.variables):
            variable.assign(store[str(i)])

    def _check_super_called(self):
        if not hasattr(self, "_lock"):
            raise RuntimeError(
                f"In optimizer '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. "
                "Go add it!"
            )

    async def apply_optimization(self, trainable_variables, reward=None, training=False):
        """Apply the backprop/optimization for each trainable variables
        that match the optimizer schema.
        """
        if not self.built:
            asyncio.get_event_loop().run_until_complete(self.build(trainable_variables))
        iteration = self._iteration.get("iteration")
        self._iteration.update({"iteration": iteration + 1})
        for variable in trainable_variables:
            if contains_schema(variable.get_schema(), self.get_schema()):
                await self.optimize(variable, reward=reward, training=training)

    async def finalize_variable_values(self, trainable_variables):
        """Finalize the optimization of the variables (cleanup/scaling etc.)."""
        for variable in trainable_variables:
            if contains_schema(variable.get_schema(), self.get_schema()):
                await self.finalize(variable)

    async def optimize(self, trainable_variable, reward=None, training=False):
        """Perform a backprop/optimization on a single variable.

        This function needs to be implemented by subclassed Optimizer.
        """
        raise NotImplementedError(
            "Optimizer subclasses must implement the `optimize()` method."
        )

    async def finalize(self, trainable_variable):
        """Finalize the optimization of the variable (cleanup/scaling etc.).

        This function needs to be implemented by subclassed Optimizer.
        """
        raise NotImplementedError(
            "Optimizer subclasses must implement the `finalize()` method."
        )

    def get_config(self):
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"<Optimizer name={self.name} description={self.description}>"
