import numpy as np
import pytensor.tensor as pt

from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.utils.constants import OBS_STATE_DIM, SHOCK_DIM
from pytensor.assumptions import assume

# Pre-industrial carbon fixes the intercept of the CO2 equation; doublings above it drive warming.
PREINDUSTRIAL_CO2_PPM = 278.0

# The rows of the observation equation, in the order the design matrix assumes.
OBSERVED = ["log_co2", "log_pop", "log_gdp", "temp", "precip", "log_emp_rate", "log_ky", "log_tfp_frontier"]

# Ordered so that no state reads one of lower index, which keeps the transition upper triangular.
# `ky_dev` reads `pop_growth`, so it sits ahead of it.
STATES = [
    "warming",
    "co2_doublings",
    "co2_growth_dev",
    "co2_drift",
    "log_tfp_frontier",
    "drift_frontier",
    "ky_dev",
    "log_pop",
    "pop_growth",
    "log_emp_rate",
    "emp_rate_growth",
    "gap",
    "gap_growth",
    "precip_dev",
    "temp_dev",
]
(
    WARMING,
    CO2,
    CO2_DEV,
    CO2_DRIFT,
    TFP,
    TFP_DRIFT,
    KY_DEV,
    POP,
    POP_GROWTH,
    EMP_RATE,
    EMP_GROWTH,
    GAP,
    GAP_GROWTH,
    PRECIP_DEV,
    TEMP_DEV,
) = range(len(STATES))

SHOCKS = ["co2", "frontier", "pop", "emp_rate", "gap", "ky", "precip", "temp"]
SHOCKED = [CO2_DEV, TFP, POP_GROWTH, EMP_GROWTH, GAP_GROWTH, KY_DEV, PRECIP_DEV, TEMP_DEV]

# The stationary deviations, initialised from their own long-run distribution. Levels and the gap
# are integrated and start where a parameter puts them.
AR_BLOCKS = ["co2", "gap_growth", "ky", "precip", "temp"]
AR_SHOCKS = [SHOCKS.index(name) for name in ("co2", "gap", "ky", "precip", "temp")]
DIFFUSE = [CO2_DEV, GAP_GROWTH, KY_DEV, PRECIP_DEV, TEMP_DEV]

LEVEL_BLOCKS = ["log_tfp_frontier", "log_pop", "log_emp_rate"]
GROWTH_BLOCKS = ["pop", "emp_rate"]
FREE_INTERCEPTS = ["temp", "precip"]
GDP_ROW = OBSERVED.index("log_gdp")
TEMP_ROW = OBSERVED.index("temp")
TFP_ROW = OBSERVED.index("log_tfp_frontier")


class ClimateEconomy(PyMCStateSpace):
    r"""
    A small open economy following a productivity frontier under a warming climate.

    Warming approaches the equilibrium the carbon stock implies geometrically, at a rate set by the
    transient climate response. Realised temperature adds a stationary deviation to it, so the two
    timescales the climate carries are separate: the forced response is slow and the internal
    variability around it is not. Temperature reaches ocean heat through a loading, precipitation
    through another, and output as a damage elasticity in log points per °C.

    The capital-output ratio is stationary, as balanced growth requires, but around a target that
    moves. Its steady state is :math:`s / (g + n + \delta)`, so it rises as population growth falls,
    and the demographic transition is the largest predictable movement in the forecast.

    Parameters
    ----------
    population_growth_mean : float
        Average population growth over the sample, in log points per year. The capital-output
        target is centred here, which leaves ``ky_mean`` the target at that rate.
    thermal_adjustment_years : float, optional
        E-folding time in years for warming to reach the equilibrium the carbon stock implies. Two
        smooth monotone series over four decades carry no information separating a lag from a
        rescaling, so this is held fixed. Default 8.0.
    """

    def __init__(self, *, population_growth_mean: float, thermal_adjustment_years: float = 8.0):
        self.population_growth_mean = population_growth_mean
        self.thermal_adjustment_years = thermal_adjustment_years
        self.thermal_persistence = float(np.exp(-1.0 / thermal_adjustment_years))

        self._build_constant_blocks()

        super().__init__(k_endog=len(OBSERVED), k_states=len(STATES), k_posdef=len(SHOCKS))

    def _build_constant_blocks(self) -> None:
        """Assemble the parts of the system that carry no parameters."""
        n_states = len(STATES)

        transition = np.zeros((n_states, n_states))
        transition[WARMING, WARMING] = self.thermal_persistence
        transition[CO2, [CO2, CO2_DEV, CO2_DRIFT]] = 1.0
        transition[CO2_DRIFT, CO2_DRIFT] = 1.0
        transition[TFP, [TFP, TFP_DRIFT]] = 1.0
        transition[TFP_DRIFT, TFP_DRIFT] = 1.0
        transition[POP, [POP, POP_GROWTH]] = 1.0
        transition[EMP_RATE, [EMP_RATE, EMP_GROWTH]] = 1.0
        transition[GAP, [GAP, GAP_GROWTH]] = 1.0
        self._transition_base = transition

        selection = np.zeros((n_states, len(SHOCKS)))
        selection[SHOCKED, range(len(SHOCKS))] = 1.0
        self._selection = selection

        design = np.zeros((len(OBSERVED), n_states))
        design[0, CO2] = np.log(2.0)
        design[1, POP] = 1.0
        design[GDP_ROW, [POP, EMP_RATE]] = 1.0
        design[4, PRECIP_DEV] = 1.0
        design[5, EMP_RATE] = 1.0
        design[6, KY_DEV] = 1.0
        design[TFP_ROW, TFP] = 1.0
        self._design_base = design

    def make_symbolic_graph(self) -> None:
        n_states = len(STATES)

        initial_level = self.make_and_register_variable("initial_level", shape=(3,))
        initial_growth = self.make_and_register_variable("initial_growth", shape=(2,))
        initial_warming = self.make_and_register_variable("initial_warming", shape=())
        initial_co2 = self.make_and_register_variable("initial_co2", shape=())
        initial_gap = self.make_and_register_variable("initial_gap", shape=())
        drift_co2 = self.make_and_register_variable("drift_co2", shape=())
        drift_frontier = self.make_and_register_variable("drift_frontier", shape=())
        drift_gap = self.make_and_register_variable("drift_gap", shape=())
        ky_mean = self.make_and_register_variable("ky_mean", shape=())
        ky_slope = self.make_and_register_variable("ky_slope", shape=())
        gdp_intercept = self.make_and_register_variable("gdp_intercept", shape=())
        climate_response = self.make_and_register_variable("climate_response", shape=())
        theta = self.make_and_register_variable("theta", shape=())
        phi = self.make_and_register_variable("phi", shape=(2,))
        rho = self.make_and_register_variable("rho", shape=(5,))
        damage_elasticity = self.make_and_register_variable("damage_elasticity", shape=())
        temp_loading = self.make_and_register_variable("temp_loading", shape=())
        precip_loading = self.make_and_register_variable("precip_loading", shape=())
        obs_intercept = self.make_and_register_variable("obs_intercept", shape=(2,))
        sigma_state = self.make_and_register_variable("sigma_state", shape=(8,))
        sigma_obs = self.make_and_register_variable("sigma_obs", shape=(8,))

        persistences = pt.set_subtensor(
            pt.zeros(n_states)[[CO2_DEV, POP_GROWTH, EMP_GROWTH, GAP_GROWTH, KY_DEV, PRECIP_DEV, TEMP_DEV]],
            pt.stack([rho[0], phi[0], phi[1], rho[1], rho[2], rho[3], rho[4]]),
        )
        transition = pt.as_tensor_variable(self._transition_base) + pt.diag(persistences)

        # How strongly warming chases the carbon stock.
        transition = pt.set_subtensor(transition[WARMING, CO2], (1.0 - self.thermal_persistence) * climate_response)

        # The capital-output target falls with population growth at `1 / (g + n + delta)`.
        reversion = 1.0 - rho[2]
        transition = pt.set_subtensor(transition[KY_DEV, POP_GROWTH], -reversion * ky_slope)
        self.ssm["transition", :, :] = assume(transition, upper_triangular=True)
        self.ssm["selection", :, :] = self._selection

        state_intercept = pt.zeros(n_states)
        # The rate at which the country falls behind the frontier it follows.
        state_intercept = pt.set_subtensor(state_intercept[GAP], drift_gap)
        # Centring the target on average population growth leaves `ky_mean` the target there.
        state_intercept = pt.set_subtensor(state_intercept[KY_DEV], reversion * ky_slope * self.population_growth_mean)
        self.ssm["state_intercept", :] = state_intercept

        # Ocean heat and output see forced warming plus its deviation; precipitation sees only the
        # forced path.
        design = pt.as_tensor_variable(self._design_base)
        design = pt.set_subtensor(
            design[:, WARMING],
            pt.stack([0.0, 0.0, -damage_elasticity, temp_loading, precip_loading, 0.0, 0.0, 0.0]),
        )
        design = pt.set_subtensor(design[GDP_ROW, TEMP_DEV], -damage_elasticity)
        design = pt.set_subtensor(design[TEMP_ROW, TEMP_DEV], temp_loading)
        productivity = 1.0 / theta
        design = pt.set_subtensor(
            design[GDP_ROW, [TFP, GAP, KY_DEV]],
            pt.stack([productivity, -productivity, (1.0 - theta) / theta]),
        )
        self.ssm["design", :, :] = design

        # `gdp_intercept` is the unit of the output series; without it the labour share carries it.
        intercept = pt.zeros(len(OBSERVED))
        intercept = pt.set_subtensor(intercept[0], np.log(PREINDUSTRIAL_CO2_PPM))
        intercept = pt.set_subtensor(intercept[[3, 4]], obs_intercept)
        intercept = pt.set_subtensor(intercept[GDP_ROW], gdp_intercept + (1.0 - theta) / theta * ky_mean)
        intercept = pt.set_subtensor(intercept[6], ky_mean)
        self.ssm["obs_intercept", :] = intercept

        initial_state = pt.zeros(n_states)
        initial_state = pt.set_subtensor(initial_state[[TFP, POP, EMP_RATE]], initial_level)
        initial_state = pt.set_subtensor(initial_state[[POP_GROWTH, EMP_GROWTH]], initial_growth)
        initial_state = pt.set_subtensor(initial_state[WARMING], initial_warming)
        initial_state = pt.set_subtensor(initial_state[CO2], initial_co2)
        initial_state = pt.set_subtensor(initial_state[CO2_DRIFT], drift_co2)
        initial_state = pt.set_subtensor(initial_state[TFP_DRIFT], drift_frontier)
        initial_state = pt.set_subtensor(initial_state[GAP], initial_gap)
        initial_state = pt.set_subtensor(
            initial_state[KY_DEV], -ky_slope * (initial_growth[0] - self.population_growth_mean)
        )
        self.ssm["initial_state", :] = initial_state

        stationary_variance = sigma_state[AR_SHOCKS] ** 2 / (1 - rho**2)
        self.ssm["initial_state_cov", :, :] = assume(
            pt.diag(pt.set_subtensor(pt.full((n_states,), 1e-6)[DIFFUSE], stationary_variance)),
            diagonal=True,
            positive_definite=True,
        )
        self.ssm["state_cov", :, :] = assume(pt.diag(sigma_state**2), diagonal=True, positive_definite=True)
        self.ssm["obs_cov", :, :] = assume(pt.diag(sigma_obs**2), diagonal=True, positive_definite=True)

    def set_parameters(self) -> tuple[Parameter, ...]:
        return (
            Parameter(name="initial_level", shape=(3,), dims=("level_block",)),
            Parameter(name="initial_growth", shape=(2,), dims=("growth_block",)),
            Parameter(name="initial_warming", shape=(), constraints="°C above pre-industrial"),
            Parameter(name="initial_co2", shape=()),
            Parameter(name="initial_gap", shape=(), constraints="log points behind"),
            Parameter(name="drift_co2", shape=()),
            Parameter(name="drift_frontier", shape=(), constraints="log points per year"),
            Parameter(name="drift_gap", shape=(), constraints="log points behind per year"),
            Parameter(name="ky_mean", shape=()),
            Parameter(name="ky_slope", shape=(), constraints="Positive"),
            Parameter(name="gdp_intercept", shape=(), constraints="log millions USD"),
            Parameter(name="climate_response", shape=(), constraints="°C per doubling"),
            Parameter(name="theta", shape=(), constraints="labour share, 0 < theta < 1"),
            Parameter(name="phi", shape=(2,), dims=("growth_block",), constraints="0 < phi < 1"),
            Parameter(name="rho", shape=(5,), dims=("ar_block",), constraints="0 < rho < 1"),
            Parameter(name="damage_elasticity", shape=(), constraints="log output per °C"),
            Parameter(name="temp_loading", shape=(), constraints="Positive"),
            Parameter(name="precip_loading", shape=()),
            Parameter(name="obs_intercept", shape=(2,), dims=("intercept",)),
            Parameter(name="sigma_state", shape=(8,), dims=(SHOCK_DIM,), constraints="Positive"),
            Parameter(name="sigma_obs", shape=(8,), dims=(OBS_STATE_DIM,), constraints="Positive"),
        )

    def set_states(self) -> tuple[State, ...]:
        return (
            *[State(name=name, observed=False) for name in STATES],
            *[State(name=name, observed=True) for name in OBSERVED],
        )

    def set_shocks(self) -> tuple[Shock, ...]:
        return tuple(Shock(name=f"{name}_innovation") for name in SHOCKS)

    def set_coords(self) -> tuple[Coord, ...]:
        return (
            *self.default_coords(),
            Coord(dimension="level_block", labels=tuple(LEVEL_BLOCKS)),
            Coord(dimension="growth_block", labels=tuple(GROWTH_BLOCKS)),
            Coord(dimension="ar_block", labels=tuple(AR_BLOCKS)),
            Coord(dimension="intercept", labels=tuple(FREE_INTERCEPTS)),
        )
