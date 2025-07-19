import dataclasses
import typing as t

from ase import units

from mlipx.abc import DynamicsModifier


@dataclasses.dataclass
class TemperatureRampModifier(DynamicsModifier):
    """Ramp the temperature from start_temperature to temperature.

    Attributes
    ----------
    start_temperature: float, optional
        temperature to start from, if None, the temperature of the thermostat is used.
    end_temperature: float
        temperature to ramp to.
    interval: int, default 1
        interval in which the temperature is changed.
    total_steps: int
        total number of steps in the simulation.

    References
    ----------
    Code taken from ipsuite/calculators/ase_md.py
    """

    end_temperature: float
    total_steps: int
    start_temperature: t.Optional[float] = None
    interval: int = 1

    def modify(self, thermostat, step):
        if self.start_temperature is None:
            if temp := getattr(thermostat, "temp", None):
                self.start_temperature = temp / units.kB
            elif temp := getattr(thermostat, "temperature", None):
                self.start_temperature = temp / units.kB
            else:
                raise AttributeError("No temperature attribute found in thermostat.")

        percentage = step / (self.total_steps - 1)
        new_temperature = (
            1 - percentage
        ) * self.start_temperature + percentage * self.end_temperature
        if step % self.interval == 0:
            if hasattr(thermostat, "set_temperature"):
                thermostat.set_temperature(temperature_K=new_temperature)
            elif hasattr(thermostat, "temperature"):
                thermostat.temperature = new_temperature * units.kB
            elif hasattr(thermostat, "temp"):
                thermostat.temp = new_temperature * units.kB
            else:
                raise AttributeError("Thermostat does not support temperature update.")




@dataclasses.dataclass
class PressureRampModifier(DynamicsModifier):
    """Ramp the pressure from start_pressure to end_pressure.

    Attributes
    ----------
    start_pressure: float, optional
        Pressure to start from (in bar). If None, attempts to use current pressure of the barostat.
    end_pressure: float
        Target pressure (in bar).
    interval: int
        Frequency (in steps) of pressure updates.
    total_steps: int
        Total number of MD steps for the ramp.
    """

    end_pressure: float
    total_steps: int
    start_pressure: t.Optional[float] = None
    interval: int = 1

    def modify(self, thermostat_barostat, step):
        if self.start_pressure is None:
            if pressure := getattr(thermostat_barostat, "pressure", None):
                self.start_pressure = pressure  # assumed already in bar
            else:
                raise AttributeError("No pressure attribute found in barostat.")

        percentage = step / (self.total_steps - 1)
        new_pressure = (
            1 - percentage
        ) * self.start_pressure + percentage * self.end_pressure

        if step % self.interval == 0:
            if hasattr(thermostat_barostat, "set_pressure"):
                thermostat_barostat.set_pressure(pressure_bar=new_pressure)
            elif hasattr(thermostat_barostat, "set_stress"):
                thermostat_barostat.set_stress(new_pressure)
            else:
                raise AttributeError("Barostat does not support pressure update via set_pressure or set_stress.")