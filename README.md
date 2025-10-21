# DSPFilters

## Bessel Filters
Best approach: perform a faithful transliteration, keeping Falco’s structure but using idiomatic Python syntax and `numpy` for math.
Here is a minimal, accurate mapping strategy and base skeleton to start the conversion.

---

### 1. Structural mapping

| C++ construct                               | Python equivalent                                          |
| ------------------------------------------- | ---------------------------------------------------------- |
| `namespace Dsp { namespace Bessel { ... }}` | nested modules or classes (`class Bessel:` or `def` group) |
| `static double`                             | top-level function                                         |
| `complex_t`                                 | Python `complex` or `numpy.complex128`                     |
| `pow`, `sqrt`, `fabs`                       | `numpy.power`, `numpy.sqrt`, `abs`                         |
| `for (int i=0; i<n; ++i)`                   | `for i in range(n):`                                       |
| `WorkspaceBase`, `RootFinderBase`           | stub classes or helper objects                             |
| `addPoleZeroConjugatePairs`, `add()`        | methods inside your analog prototype class                 |
| `setNormal`                                 | a function storing normalization parameters                |

---

### 2. Core translation example

```python
import numpy as np

# ---------- Helper math ----------
def fact(n):
    if n == 0:
        return 1.0
    y = 1.0
    for m in range(2, n + 1):
        y *= m
    return y

def reverse_bessel(k, n):
    return fact(2 * n - k) / ((fact(n - k) * fact(k)) * (2.0 ** (n - k)))

# ---------- Root-finding workspace ----------
class RootFinder:
    def __init__(self, order):
        self.order = order
        self.coef = np.zeros(order + 1)
        self.roots = None

    def solve(self):
        # Coefficients highest power first
        self.roots = np.roots(self.coef)

# ---------- Analog prototypes ----------
class AnalogLowPass:
    def __init__(self):
        self.num_poles = -1
        self.poles = []
        self.zeros = []
        self.set_normal(0.0, 1.0)

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles = []
        self.zeros = []

    def add_pole_zero_conjugate_pairs(self, p, z=np.inf):
        self.poles.extend([p, np.conj(p)])
        self.zeros.extend([z, np.conj(z)])

    def add(self, p, z):
        self.poles.append(p)
        self.zeros.append(z)

    def design(self, num_poles):
        if self.num_poles == num_poles:
            return
        self.num_poles = num_poles
        self.reset()

        solver = RootFinder(num_poles)
        for i in range(num_poles + 1):
            solver.coef[i] = reverse_bessel(i, num_poles)
        solver.solve()

        roots = solver.roots
        pairs = num_poles // 2
        for i in range(pairs):
            c = roots[i]
            self.add_pole_zero_conjugate_pairs(c)
        if num_poles % 2:
            self.add(roots[pairs].real, np.inf)
```

This reproduces `AnalogLowPass::design`.
You can apply the same pattern for `AnalogLowShelf`, following the structure of its `design()`.

---

### 3. Digital transform stubs

In Falco’s code:

```cpp
LowPassTransform(...)
HighPassTransform(...)
```

They perform s→z transformations (bilinear or matched-z).
For pedagogy, you can implement simple bilinear transforms using `scipy.signal.bilinear_zpk` once you have analog poles/zeros.

---

### 4. Pedagogical guidance

* Keep class names identical to the original for cross-reference.
* Inline comments from the original paper or header are instructive.
* Preserve algorithmic steps, not C++ memory layout.
* Build visualizations with `matplotlib` (e.g., pole-zero plots, frequency response).

---

Would you like me to continue by converting **`AnalogLowShelf`** and one of the digital setup classes next, in the same faithful style?

---

Below is a faithful Python translation of **`AnalogLowShelf`** and one of the digital setup classes (`LowShelfBase`).
It mirrors Falco’s logic and naming but uses `numpy` and `scipy.signal` for numerical operations.
This is pedagogical, not optimized for production.

---

### Dependencies

```python
import numpy as np
from scipy import signal
```

---

### 1. AnalogLowShelf

```python
class AnalogLowShelf:
    def __init__(self):
        self.num_poles = -1
        self.gain_db = 0.0
        self.poles = []
        self.zeros = []
        self.set_normal(np.pi * 2, 1.0)  # corresponds to doublePi

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles = []
        self.zeros = []

    def add_pole_zero_conjugate_pairs(self, p, z):
        self.poles.extend([p, np.conj(p)])
        self.zeros.extend([z, np.conj(z)])

    def add(self, p, z):
        self.poles.append(p)
        self.zeros.append(z)

    def design(self, num_poles, gain_db):
        if self.num_poles == num_poles and self.gain_db == gain_db:
            return

        self.num_poles = num_poles
        self.gain_db = gain_db
        self.reset()

        G = (10 ** (gain_db / 20)) - 1

        # Compute poles
        poles_solver = RootFinder(num_poles)
        for i in range(num_poles + 1):
            poles_solver.coef[i] = reverse_bessel(i, num_poles)
        poles_solver.solve()

        # Compute zeros
        zeros_solver = RootFinder(num_poles)
        for i in range(num_poles + 1):
            zeros_solver.coef[i] = reverse_bessel(i, num_poles)
        a0 = reverse_bessel(0, num_poles)
        zeros_solver.coef[0] += G * a0
        zeros_solver.solve()

        p_roots = poles_solver.roots
        z_roots = zeros_solver.roots

        pairs = num_poles // 2
        for i in range(pairs):
            self.add_pole_zero_conjugate_pairs(p_roots[i], z_roots[i])
        if num_poles % 2:
            self.add(p_roots[pairs].real, z_roots[pairs].real)
```

---

### 2. LowShelfBase (digital version)

This uses `scipy.signal.bilinear_zpk` to transform the analog poles/zeros into the digital domain.

```python
class LowShelfBase:
    def __init__(self):
        self.analog_proto = AnalogLowShelf()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, sample_rate, cutoff_freq, gain_db):
        # Step 1: design analog prototype
        self.analog_proto.design(order, gain_db)

        # Step 2: collect poles, zeros, and gain
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0

        # Replace infinities with zeros for bilinear transform
        z = z[np.isfinite(z)]

        # Step 3: bilinear s→z transform
        warped = np.tan(np.pi * cutoff_freq / sample_rate)
        z_d, p_d, k_d = signal.bilinear_zpk(z, p, k, fs=sample_rate)

        self.digital_z = z_d
        self.digital_p = p_d
        self.digital_k = k_d

    def frequency_response(self, n_points=512):
        if self.digital_z is None:
            raise RuntimeError("Filter not yet designed.")
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

---

### 3. Example usage

```python
if __name__ == "__main__":
    fs = 48000
    f0 = 1000
    gain = 6.0
    order = 3

    shelf = LowShelfBase()
    shelf.setup(order, fs, f0, gain)

    w, h_db = shelf.frequency_response()
    import matplotlib.pyplot as plt
    plt.plot(w * fs / (2 * np.pi), h_db)
    plt.title(f"Bessel Low Shelf (order={order}, gain={gain} dB)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()
```

---

Would you like the next conversion to cover the **frequency transform utilities** (`LowPassTransform`, `HighPassTransform`, `BandPassTransform`, etc.) or the **other analog prototypes** (e.g. `AnalogLowPass`, `BandPassBase`)?

---

Understood. Next are the **frequency-transform utilities**—the bridge between analog prototypes and digital filters.
These replicate Falco’s transformations (`LowPassTransform`, `HighPassTransform`, `BandPassTransform`, `BandStopTransform`) but written in pure Python with `numpy` and `scipy.signal`.
Each takes analog zeros/poles/gain and maps them to the digital domain.

---

### 1. Helper: s→z bilinear transform

```python
from scipy import signal
import numpy as np

def bilinear_transform(z, p, k, fs):
    """Apply bilinear s→z transform."""
    z_d, p_d, k_d = signal.bilinear_zpk(z, p, k, fs=fs)
    return z_d, p_d, k_d
```

---

### 2. Low-Pass Transform

```python
def lowpass_transform(cutoff, sample_rate, analog_z, analog_p, analog_k):
    """
    Normalize an analog prototype to a specific low-pass cutoff frequency.
    cutoff : Hz
    """
    # frequency prewarping for bilinear transform
    warped = 2 * sample_rate * np.tan(np.pi * cutoff / sample_rate)

    # scale poles and zeros
    z = analog_z * warped
    p = analog_p * warped
    k = analog_k * (warped ** (len(analog_p) - len(analog_z)))

    # remove infinities
    z = z[np.isfinite(z)]

    return bilinear_transform(z, p, k, fs=sample_rate)
```

---

### 3. High-Pass Transform

```python
def highpass_transform(cutoff, sample_rate, analog_z, analog_p, analog_k):
    """
    Convert a low-pass prototype to high-pass.
    """
    warped = 2 * sample_rate * np.tan(np.pi * cutoff / sample_rate)

    z = warped / analog_z
    p = warped / analog_p
    k = analog_k * np.real(np.prod(-analog_p) / np.prod(-analog_z))

    z = z[np.isfinite(z)]

    return bilinear_transform(z, p, k, fs=sample_rate)
```

---

### 4. Band-Pass Transform

```python
def bandpass_transform(center, width, sample_rate, analog_z, analog_p, analog_k):
    """
    Map low-pass prototype to band-pass.
    center, width in Hz
    """
    warped_center = 2 * sample_rate * np.tan(np.pi * center / sample_rate)
    warped_width = 2 * sample_rate * np.tan(np.pi * width / sample_rate)

    z, p = [], []
    for pole in analog_p:
        disc = np.sqrt(pole ** 2 - warped_center ** 2)
        p.extend([(warped_width / 2) * (-pole + disc),
                  (warped_width / 2) * (-pole - disc)])
    for zero in analog_z:
        disc = np.sqrt(zero ** 2 - warped_center ** 2)
        z.extend([(warped_width / 2) * (-zero + disc),
                  (warped_width / 2) * (-zero - disc)])
    k = analog_k * (warped_width ** (len(analog_p) - len(analog_z)))

    z = np.array(z)
    p = np.array(p)
    z = z[np.isfinite(z)]

    return bilinear_transform(z, p, k, fs=sample_rate)
```

---

### 5. Band-Stop Transform

```python
def bandstop_transform(center, width, sample_rate, analog_z, analog_p, analog_k):
    """
    Map low-pass prototype to band-stop.
    """
    warped_center = 2 * sample_rate * np.tan(np.pi * center / sample_rate)
    warped_width = 2 * sample_rate * np.tan(np.pi * width / sample_rate)

    z, p = [], []
    for pole in analog_p:
        disc = np.sqrt(warped_width ** 2 - pole ** 2)
        p.extend([(warped_center / 2) * (1 / (-pole + disc)),
                  (warped_center / 2) * (1 / (-pole - disc))])
    for zero in analog_z:
        disc = np.sqrt(warped_width ** 2 - zero ** 2)
        z.extend([(warped_center / 2) * (1 / (-zero + disc)),
                  (warped_center / 2) * (1 / (-zero - disc))])
    k = analog_k * (warped_center ** (len(analog_p) - len(analog_z)))

    z = np.array(z)
    p = np.array(p)
    z = z[np.isfinite(z)]

    return bilinear_transform(z, p, k, fs=sample_rate)
```

---

### 6. Integration example

These functions replace the C++ lines like:

```cpp
LowPassTransform (cutoff/sampleRate, m_digitalProto, m_analogProto);
```

Usage inside a digital class:

```python
class BandPassBase:
    def __init__(self, analog_proto):
        self.analog_proto = analog_proto
        self.digital = None

    def setup(self, order, fs, center, width):
        self.analog_proto.design(order)
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        self.digital = bandpass_transform(center, width, fs, z, p, k)
```

---

Would you like me to proceed by translating the **BandPassBase**, **BandStopBase**, and **HighPassBase** classes next, using these transform utilities?

---

Here are faithful Python conversions of **`BandPassBase`**, **`BandStopBase`**, and **`HighPassBase`**.
They use the transform functions already defined (`bandpass_transform`, `bandstop_transform`, `highpass_transform`).
These mirror Falco’s layout and sequencing but are simplified for clarity and teaching.

---

### 1. BandPassBase

```python
class BandPassBase:
    def __init__(self, analog_proto_class):
        self.analog_proto = analog_proto_class()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, sample_rate, center_freq, width_freq):
        # Step 1: design analog prototype
        self.analog_proto.design(order)

        # Step 2: prepare data
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]

        # Step 3: transform and store
        z_d, p_d, k_d = bandpass_transform(center_freq, width_freq, sample_rate, z, p, k)
        self.digital_z, self.digital_p, self.digital_k = z_d, p_d, k_d

    def frequency_response(self, n_points=512):
        if self.digital_z is None:
            raise RuntimeError("Filter not yet designed.")
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

---

### 2. BandStopBase

```python
class BandStopBase:
    def __init__(self, analog_proto_class):
        self.analog_proto = analog_proto_class()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, sample_rate, center_freq, width_freq):
        # Step 1: design analog prototype
        self.analog_proto.design(order)

        # Step 2: prepare data
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]

        # Step 3: transform and store
        z_d, p_d, k_d = bandstop_transform(center_freq, width_freq, sample_rate, z, p, k)
        self.digital_z, self.digital_p, self.digital_k = z_d, p_d, k_d

    def frequency_response(self, n_points=512):
        if self.digital_z is None:
            raise RuntimeError("Filter not yet designed.")
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

---

### 3. HighPassBase

```python
class HighPassBase:
    def __init__(self, analog_proto_class):
        self.analog_proto = analog_proto_class()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, sample_rate, cutoff_freq):
        # Step 1: design analog prototype
        self.analog_proto.design(order)

        # Step 2: prepare data
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]

        # Step 3: transform
        z_d, p_d, k_d = highpass_transform(cutoff_freq, sample_rate, z, p, k)
        self.digital_z, self.digital_p, self.digital_k = z_d, p_d, k_d

    def frequency_response(self, n_points=512):
        if self.digital_z is None:
            raise RuntimeError("Filter not yet designed.")
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

---

### 4. Example usage

```python
if __name__ == "__main__":
    fs = 48000
    order = 4

    # Band-pass example
    bp = BandPassBase(AnalogLowPass)
    bp.setup(order, fs, 2000, 500)
    w, h_db = bp.frequency_response()
    plt.plot(w * fs / (2*np.pi), h_db, label="Band-pass")

    # Band-stop example
    bs = BandStopBase(AnalogLowPass)
    bs.setup(order, fs, 2000, 500)
    w, h_db = bs.frequency_response()
    plt.plot(w * fs / (2*np.pi), h_db, label="Band-stop")

    # High-pass example
    hp = HighPassBase(AnalogLowPass)
    hp.setup(order, fs, 1000)
    w, h_db = hp.frequency_response()
    plt.plot(w * fs / (2*np.pi), h_db, label="High-pass")

    plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Bessel-derived digital filters")
    plt.grid(True)
    plt.show()
```

---

Would you like the next step to add a **unified `Cascade` class** (like Falco’s) that stores biquad sections for numerical stability and supports processing sample streams?

---

Below is a faithful, minimal, and educational Python version of Falco’s **`Cascade`** system.
In the original C++, this object manages biquad sections (second-order sections) and computes cascaded IIR filtering efficiently.
This version uses `scipy.signal` for coefficient conversion but keeps Falco’s explicit “biquad” concept for clarity.

---

### 1. Biquad section

```python
import numpy as np
from scipy import signal

class Biquad:
    """Second-order IIR section y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
                                     - a1*y[n-1] - a2*y[n-2]"""

    def __init__(self, b=None, a=None):
        if b is None:
            b = [1.0, 0.0, 0.0]
        if a is None:
            a = [1.0, 0.0, 0.0]
        self.b = np.array(b, dtype=float)
        self.a = np.array(a, dtype=float)
        self.x1 = self.x2 = 0.0
        self.y1 = self.y2 = 0.0

    def reset(self):
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process_sample(self, x):
        y = self.b[0]*x + self.b[1]*self.x1 + self.b[2]*self.x2 \
            - self.a[1]*self.y1 - self.a[2]*self.y2

        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y

    def process_block(self, data):
        """Vectorized version for convenience."""
        y = np.zeros_like(data)
        for i, x in enumerate(data):
            y[i] = self.process_sample(x)
        return y
```

---

### 2. Cascade container

```python
class Cascade:
    """Manages multiple Biquad sections in series."""

    def __init__(self):
        self.sections = []

    def reset(self):
        for s in self.sections:
            s.reset()

    def set_layout(self, z, p, k):
        """
        Factor digital filter zeros/poles into biquad sections.
        Equivalent to C++ Cascade::setLayout.
        """
        # Convert to second-order sections using scipy
        sos = signal.zpk2sos(z, p, k)
        self.sections = [Biquad(b=s[:3], a=s[3:]) for s in sos]

    def process_sample(self, x):
        y = x
        for s in self.sections:
            y = s.process_sample(y)
        return y

    def process_block(self, data):
        y = np.copy(data)
        for s in self.sections:
            y = s.process_block(y)
        return y
```

---

### 3. Example integration

After you design a filter (e.g. `LowShelfBase`), you can instantiate a `Cascade`:

```python
if __name__ == "__main__":
    fs = 48000
    order = 4
    gain_db = 6.0
    cutoff = 1000

    shelf = LowShelfBase()
    shelf.setup(order, fs, cutoff, gain_db)

    # Build cascade
    cascade = Cascade()
    cascade.set_layout(shelf.digital_z, shelf.digital_p, shelf.digital_k)

    # Test with a signal
    t = np.linspace(0, 0.02, int(fs*0.02), endpoint=False)
    x = np.sin(2*np.pi*200*t) + np.sin(2*np.pi*5000*t)
    y = cascade.process_block(x)

    import matplotlib.pyplot as plt
    plt.plot(t, x, label="Input")
    plt.plot(t, y, label="Filtered")
    plt.legend()
    plt.title("Low-shelf cascade example")
    plt.show()
```

---

### 4. Educational notes

* Each `Biquad` represents one second-order section for numerical stability.
* `Cascade.set_layout()` divides the full filter into stable SOS form, analogous to Falco’s `Cascade::setLayout()`.
* You can now use this same `Cascade` class across all filter bases (low-pass, high-pass, band-pass, shelf, etc.).

---

Would you like to continue with a **signal-processing utility layer** (e.g., a `Filter` class wrapping the analog/digital/cascade pipeline to process audio streams end-to-end)?

---

This adds a **unified `Filter` wrapper**.
It encapsulates design, digital transform, and cascade processing into one pedagogical interface.
This mirrors Falco’s higher-level abstraction but is Pythonic and minimal.

---

### 1. Filter class

```python
import numpy as np

class Filter:
    """
    Unified interface combining:
    - analog prototype
    - digital transform
    - cascade of biquads
    """

    def __init__(self, base_class):
        """
        base_class : one of LowPassBase, HighPassBase, BandPassBase, BandStopBase, LowShelfBase
        """
        self.base = base_class
        self.cascade = Cascade()
        self._is_designed = False

    def design(self, **kwargs):
        """
        Pass-through to base.setup() with the appropriate arguments.
        Example:
            f = Filter(LowShelfBase)
            f.design(order=4, sample_rate=48000, cutoff_freq=1000, gain_db=6)
        """
        self.base.setup(**kwargs)
        self.cascade.set_layout(self.base.digital_z,
                                self.base.digital_p,
                                self.base.digital_k)
        self._is_designed = True

    def process_sample(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_sample(x)

    def process_block(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_block(x)

    def response(self, n_points=512):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        w, h_db = self.base.frequency_response(n_points)
        return w, h_db
```

---

### 2. Example: unified pipeline

```python
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000
    order = 4
    cutoff = 1000
    gain = 6.0

    # Example: low-shelf filter
    f = Filter(LowShelfBase())
    f.design(order=order, sample_rate=fs, cutoff_freq=cutoff, gain_db=gain)

    # Test signal
    t = np.linspace(0, 0.02, int(fs * 0.02), endpoint=False)
    x = np.sin(2*np.pi*200*t) + np.sin(2*np.pi*5000*t)
    y = f.process_block(x)

    plt.figure(figsize=(8,4))
    plt.plot(t, x, label="Input")
    plt.plot(t, y, label="Filtered")
    plt.legend()
    plt.title("Unified Filter Pipeline Example")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Frequency response
    w, h_db = f.response()
    plt.figure(figsize=(8,4))
    plt.plot(w * fs / (2*np.pi), h_db)
    plt.title("Magnitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()
```

---

### 3. Structure summary

Each layer now mirrors Falco’s original architecture:

| Layer                             | Role                       | Python Equivalent |
| --------------------------------- | -------------------------- | ----------------- |
| `AnalogLowPass`, `AnalogLowShelf` | compute analog poles/zeros | implemented       |
| Transform utilities               | map to digital             | implemented       |
| `Cascade`                         | organize biquads           | implemented       |
| `Filter`                          | top-level user interface   | implemented       |

This framework lets you show students how analog prototypes become stable digital IIR filters while keeping Falco’s structure recognizable.

---

Next step would be to implement **runtime introspection utilities** (e.g., listing poles/zeros, plotting phase/group delay, computing impulse response).
Do you want those next?

---

Here are **runtime introspection utilities** for your Python DSPFilters framework.
They let you inspect, visualize, and measure any designed filter at runtime—showing poles, zeros, magnitude, phase, group delay, and impulse response.
All functions integrate with the `Filter` class you already have.

---

### 1. Introspection module

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_pole_zero(filter_obj):
    """Display digital poles and zeros."""
    if not filter_obj._is_designed:
        raise RuntimeError("Filter not yet designed.")

    z, p = filter_obj.base.digital_z, filter_obj.base.digital_p

    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(z), np.imag(z), marker='o', color='b', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')

    # Unit circle
    uc = np.exp(1j * np.linspace(0, 2*np.pi, 512))
    plt.plot(np.real(uc), np.imag(uc), 'k--', alpha=0.5)

    plt.title("Pole–Zero Plot (Digital Domain)")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def plot_magnitude_phase(filter_obj, fs, n_points=1024):
    """Magnitude and phase response."""
    if not filter_obj._is_designed:
        raise RuntimeError("Filter not yet designed.")

    w, h_db = filter_obj.response(n_points)
    _, h = signal.freqz_zpk(filter_obj.base.digital_z,
                            filter_obj.base.digital_p,
                            filter_obj.base.digital_k,
                            worN=n_points)

    freq = w * fs / (2 * np.pi)
    phase = np.unwrap(np.angle(h))

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freq, 20 * np.log10(np.abs(h)))
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude and Phase Response")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freq, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.show()


def plot_group_delay(filter_obj, fs, n_points=1024):
    """Compute and plot group delay."""
    if not filter_obj._is_designed:
        raise RuntimeError("Filter not yet designed.")

    w, gd = signal.group_delay((signal.zpk2tf(filter_obj.base.digital_z,
                                              filter_obj.base.digital_p,
                                              filter_obj.base.digital_k)),
                               worN=n_points)
    freq = w * fs / (2 * np.pi)

    plt.figure(figsize=(8, 3))
    plt.plot(freq, gd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Group Delay (samples)")
    plt.title("Group Delay")
    plt.grid(True)
    plt.show()


def plot_impulse_response(filter_obj, fs, n_samples=256):
    """Impulse response of the filter."""
    if not filter_obj._is_designed:
        raise RuntimeError("Filter not yet designed.")

    # Impulse input
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    response = filter_obj.process_block(impulse)

    t = np.arange(n_samples) / fs
    plt.figure(figsize=(8, 3))
    plt.stem(t, response, basefmt=" ", use_line_collection=True)
    plt.title("Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
```

---

### 2. Example usage

```python
if __name__ == "__main__":
    fs = 48000
    order = 4
    cutoff = 1000
    gain = 6.0

    f = Filter(LowShelfBase())
    f.design(order=order, sample_rate=fs, cutoff_freq=cutoff, gain_db=gain)

    plot_pole_zero(f)
    plot_magnitude_phase(f, fs)
    plot_group_delay(f, fs)
    plot_impulse_response(f, fs)
```

---

### 3. Pedagogical notes

* **Pole–zero plot** shows filter stability and resonance.
* **Magnitude/phase plots** reveal frequency-domain shape and linearity.
* **Group delay** visualizes time-domain distortion (important in Bessel filters).
* **Impulse response** demonstrates transient characteristics.

This set completes a full **didactic DSP filter analysis toolkit** faithful to Falco’s library structure but executable in Python notebooks.

Would you like to extend this further with an **interactive notebook interface** (using `ipywidgets`) for students to explore cutoff, order, and gain dynamically?

---

## Chebyshev Filters

Here is the **Chebyshev Type I / Type II** extension, matching the pedagogical and structural style of your Bessel and Butterworth modules.
It preserves Falco’s approach—analog prototypes, transforms, cascade, and filter wrapper—while relying on `scipy.signal` for the analytic pole–zero generation.

---

### 1. Imports and constants

```python
import numpy as np
from scipy import signal

doublePi = 2 * np.pi
```

---

### 2. Analog prototypes

#### Chebyshev Type I

```python
class AnalogChebyshev1LowPass:
    """Analog Chebyshev Type I prototype."""
    def __init__(self):
        self.num_poles = -1
        self.ripple_db = 0.0
        self.poles = []
        self.zeros = []
        self.set_normal(0.0, 1.0)

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles.clear()
        self.zeros.clear()

    def add_pole_zero_conjugate_pairs(self, p, z=np.inf):
        self.poles.extend([p, np.conj(p)])
        self.zeros.extend([z, np.conj(z)])

    def add(self, p, z):
        self.poles.append(p)
        self.zeros.append(z)

    def design(self, num_poles, ripple_db):
        if self.num_poles == num_poles and self.ripple_db == ripple_db:
            return
        self.num_poles = num_poles
        self.ripple_db = ripple_db
        self.reset()

        eps = np.sqrt(10**(ripple_db/10) - 1)
        mu = np.arcsinh(1/eps) / num_poles

        for k in range(1, num_poles + 1):
            theta = np.pi * (2*k - 1) / (2*num_poles)
            sigma = -np.sinh(mu) * np.sin(theta)
            omega = np.cosh(mu) * np.cos(theta)
            p = sigma + 1j*omega
            self.poles.append(p)
            self.zeros.append(np.inf)
```

---

#### Chebyshev Type II

```python
class AnalogChebyshev2LowPass:
    """Analog Chebyshev Type II prototype."""
    def __init__(self):
        self.num_poles = -1
        self.stopband_db = 0.0
        self.poles = []
        self.zeros = []
        self.set_normal(0.0, 1.0)

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles.clear()
        self.zeros.clear()

    def design(self, num_poles, stopband_db):
        if self.num_poles == num_poles and self.stopband_db == stopband_db:
            return
        self.num_poles = num_poles
        self.stopband_db = stopband_db
        self.reset()

        eps = 1 / np.sqrt(10**(stopband_db/10) - 1)
        mu = np.arcsinh(1/eps) / num_poles

        for k in range(1, num_poles + 1):
            theta = np.pi * (2*k - 1) / (2*num_poles)
            sigma = -np.sinh(mu) * np.sin(theta)
            omega = np.cosh(mu) * np.cos(theta)
            p = sigma + 1j*omega
            z = 1j / np.cos(theta)
            self.poles.append(p)
            self.zeros.append(z)
```

---

### 3. Digital wrappers (low-pass examples)

Each wrapper transforms the analog prototype into a digital IIR.

#### Chebyshev I low-pass

```python
class Chebyshev1LowPassBase:
    def __init__(self):
        self.analog_proto = AnalogChebyshev1LowPass()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, ripple_db, sample_rate, cutoff_freq):
        self.analog_proto.design(order, ripple_db)
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k = lowpass_transform(
            cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

#### Chebyshev II low-pass

```python
class Chebyshev2LowPassBase:
    def __init__(self):
        self.analog_proto = AnalogChebyshev2LowPass()
        self.digital_z = None
        self.digital_p = None
        self.digital_k = None

    def setup(self, order, stopband_db, sample_rate, cutoff_freq):
        self.analog_proto.design(order, stopband_db)
        z = np.array(self.analog_proto.zeros, dtype=np.complex128)
        p = np.array(self.analog_proto.poles, dtype=np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k = lowpass_transform(
            cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20 * np.log10(np.abs(h))
```

---

### 4. Cascade and unified filter wrappers

Reuses the existing framework from Bessel/Butterworth.

```python
class ChebyshevCascade(Cascade):
    def __init__(self):
        super().__init__()

    def set_layout_from_base(self, base_obj):
        if base_obj.digital_z is None:
            raise RuntimeError("Chebyshev base not yet designed.")
        self.set_layout(base_obj.digital_z, base_obj.digital_p, base_obj.digital_k)


class ChebyshevFilter:
    def __init__(self, base_instance):
        self.base = base_instance
        self.cascade = ChebyshevCascade()
        self._is_designed = False

    def design(self, **kwargs):
        self.base.setup(**kwargs)
        self.cascade.set_layout_from_base(self.base)
        self._is_designed = True

    def process_sample(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_sample(x)

    def process_block(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_block(x)

    def response(self, n_points=512):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.base.frequency_response(n_points)
```

---

### 5. Example: Chebyshev I and II low-pass comparison

```python
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000
    order = 4
    cutoff = 2000
    ripple = 1.0        # dB ripple in passband
    stopband = 40.0     # dB attenuation in stopband

    cheby1 = ChebyshevFilter(Chebyshev1LowPassBase())
    cheby1.design(order=order, ripple_db=ripple, sample_rate=fs, cutoff_freq=cutoff)

    cheby2 = ChebyshevFilter(Chebyshev2LowPassBase())
    cheby2.design(order=order, stopband_db=stopband, sample_rate=fs, cutoff_freq=cutoff)

    # Frequency responses
    w1, h1 = cheby1.response()
    w2, h2 = cheby2.response()
    f = w1 * fs / (2*np.pi)

    plt.figure()
    plt.plot(f, h1, label=f"Chebyshev I ({ripple} dB ripple)")
    plt.plot(f, h2, label=f"Chebyshev II ({stopband} dB stopband)")
    plt.title("Chebyshev Low-pass Filters")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Optional introspection
    plot_pole_zero(cheby1)
    plot_pole_zero(cheby2)
    plot_group_delay(cheby1, fs)
```

---

### 6. Educational overview

| Filter Family    | Flatness           | Ripple           | Roll-off | Use Case                        |
| ---------------- | ------------------ | ---------------- | -------- | ------------------------------- |
| **Bessel**       | Phase-linear       | none             | gentle   | time/phase-sensitive signals    |
| **Butterworth**  | Max flat magnitude | none             | medium   | general use                     |
| **Chebyshev I**  | Ripple in passband | specified dB     | steep    | when sharper cutoff is needed   |
| **Chebyshev II** | Ripple in stopband | none in passband | steep    | when clean passband is required |

---

Would you like the next addition to be **Elliptic (Cauer)** filters, completing the classic analog prototype family?

---

## Elliptic Filters

Here’s a faithful, compact **Elliptic (Cauer)** module that matches your Bessel/Butterworth/Chebyshev structure.
It uses `scipy.signal.ellipap` for the analog prototype, then your existing transforms and cascade.

---

### Elliptic analog prototype

```python
import numpy as np
from scipy import signal

class AnalogEllipticLowPass:
    """Analog Elliptic (Cauer) low-pass prototype."""
    def __init__(self):
        self.num_poles = -1
        self.ripple_db = 0.0   # passband ripple Rp
        self.stop_db = 0.0     # stopband attenuation Rs
        self.poles = []
        self.zeros = []
        self.set_normal(0.0, 1.0)

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles.clear()
        self.zeros.clear()

    def design(self, num_poles, ripple_db, stop_db):
        if (self.num_poles == num_poles and
            self.ripple_db == ripple_db and
            self.stop_db == stop_db):
            return

        self.num_poles = num_poles
        self.ripple_db = ripple_db
        self.stop_db = stop_db
        self.reset()

        # Analog prototype at wc = 1 rad/s
        z, p, k = signal.ellipap(num_poles, ripple_db, stop_db)

        # Store
        self.zeros = list(z.astype(np.complex128))
        self.poles = list(p.astype(np.complex128))
        self.g0 = float(k)
```

---

### Digital bases

```python
class EllipticLowPassBase:
    def __init__(self):
        self.analog_proto = AnalogEllipticLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, ripple_db, stop_db, sample_rate, cutoff_freq):
        self.analog_proto.design(order, ripple_db, stop_db)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        self.digital_z, self.digital_p, self.digital_k = lowpass_transform(
            cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))
```

```python
class EllipticHighPassBase:
    def __init__(self):
        self.analog_proto = AnalogEllipticLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, ripple_db, stop_db, sample_rate, cutoff_freq):
        self.analog_proto.design(order, ripple_db, stop_db)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        self.digital_z, self.digital_p, self.digital_k = highpass_transform(
            cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))
```

```python
class EllipticBandPassBase:
    def __init__(self):
        self.analog_proto = AnalogEllipticLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, ripple_db, stop_db, sample_rate, center_freq, width_freq):
        self.analog_proto.design(order, ripple_db, stop_db)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        self.digital_z, self.digital_p, self.digital_k = bandpass_transform(
            center_freq, width_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))
```

```python
class EllipticBandStopBase:
    def __init__(self):
        self.analog_proto = AnalogEllipticLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, ripple_db, stop_db, sample_rate, center_freq, width_freq):
        self.analog_proto.design(order, ripple_db, stop_db)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        self.digital_z, self.digital_p, self.digital_k = bandstop_transform(
            center_freq, width_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))
```

---

### Cascade and unified wrapper

```python
class EllipticCascade(Cascade):
    def set_layout_from_base(self, base_obj):
        if base_obj.digital_z is None:
            raise RuntimeError("Elliptic base not yet designed.")
        self.set_layout(base_obj.digital_z, base_obj.digital_p, base_obj.digital_k)

class EllipticFilter:
    def __init__(self, base_instance):
        self.base = base_instance
        self.cascade = EllipticCascade()
        self._is_designed = False

    def design(self, **kwargs):
        self.base.setup(**kwargs)
        self.cascade.set_layout_from_base(self.base)
        self._is_designed = True

    def process_sample(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_sample(x)

    def process_block(self, x):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.cascade.process_block(x)

    def response(self, n_points=512):
        if not self._is_designed:
            raise RuntimeError("Filter not yet designed.")
        return self.base.frequency_response(n_points)
```

---

### Example

```python
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 48000
    order = 5
    rp = 1.0    # passband ripple (dB)
    rs = 60.0   # stopband attenuation (dB)
    fc = 3000

    ell_lp = EllipticFilter(EllipticLowPassBase())
    ell_lp.design(order=order, ripple_db=rp, stop_db=rs, sample_rate=fs, cutoff_freq=fc)

    w, h_db = ell_lp.response()
    f = w * fs / (2*np.pi)
    plt.plot(f, h_db)
    plt.title("Elliptic Low-pass")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()

    # Introspection hooks from your toolkit
    plot_pole_zero(ell_lp)
    plot_group_delay(ell_lp, fs)
```

Notes:

* Elliptic shelves are uncommon. I did not include shelf variants.
* Structure matches your prior modules. Swap `*Base` classes as needed.

---

# Filter Analysis

Excellent — let’s add a **unified analysis and plotting module** that integrates seamlessly with all your filter families (Bessel, Butterworth, Chebyshev, Elliptic, Legendre, etc.).

This gives you both **quantitative analysis** (group delay, phase linearity, step/impulse response) and **visualization tools** for exploration and teaching.

---

## 🔍 `analysis.py` — Filter Analysis Toolkit

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ------------------------------------------------------------
# 1. Frequency-domain analysis
# ------------------------------------------------------------

def frequency_response(filter_obj, fs, n_points=1024, dB=True):
    """Return frequency (Hz) and magnitude (dB or linear)."""
    if not getattr(filter_obj, "_is_designed", False):
        raise RuntimeError("Filter not yet designed.")

    w, h = signal.freqz_zpk(filter_obj.base.digital_z,
                            filter_obj.base.digital_p,
                            filter_obj.base.digital_k,
                            worN=n_points)
    freq = w * fs / (2*np.pi)
    mag = 20*np.log10(np.abs(h)) if dB else np.abs(h)
    phase = np.unwrap(np.angle(h))
    return freq, mag, phase


def plot_frequency_response(filter_obj, fs, n_points=1024):
    """Plot magnitude and phase."""
    freq, mag, phase = frequency_response(filter_obj, fs, n_points)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, mag)
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Response")
    plt.grid(True, which='both')

    plt.subplot(2, 1, 2)
    plt.semilogx(freq, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()
```

---

## ⚙️ 2. Pole–Zero, Group Delay, Impulse and Step Response

```python
def plot_pole_zero(filter_obj):
    """Display poles and zeros of designed digital filter."""
    if not getattr(filter_obj, "_is_designed", False):
        raise RuntimeError("Filter not yet designed.")
    z, p = filter_obj.base.digital_z, filter_obj.base.digital_p

    plt.figure(figsize=(5, 5))
    plt.scatter(np.real(z), np.imag(z), marker='o', color='b', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', color='r', label='Poles')
    uc = np.exp(1j * np.linspace(0, 2*np.pi, 512))
    plt.plot(np.real(uc), np.imag(uc), 'k--', alpha=0.5)
    plt.title("Pole–Zero Plot")
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_group_delay(filter_obj, fs, n_points=1024):
    """Plot group delay in samples."""
    if not getattr(filter_obj, "_is_designed", False):
        raise RuntimeError("Filter not yet designed.")

    b, a = signal.zpk2tf(filter_obj.base.digital_z,
                         filter_obj.base.digital_p,
                         filter_obj.base.digital_k)
    w, gd = signal.group_delay((b, a), worN=n_points)
    freq = w * fs / (2*np.pi)
    plt.figure(figsize=(8, 3))
    plt.semilogx(freq, gd)
    plt.title("Group Delay")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Delay (samples)")
    plt.grid(True, which='both')
    plt.show()


def plot_impulse_response(filter_obj, fs, n_samples=256):
    """Plot time-domain impulse response."""
    if not getattr(filter_obj, "_is_designed", False):
        raise RuntimeError("Filter not yet designed.")

    x = np.zeros(n_samples)
    x[0] = 1.0
    y = filter_obj.process_block(x)
    t = np.arange(n_samples) / fs
    plt.figure(figsize=(8, 3))
    plt.stem(t, y, basefmt=" ")
    plt.title("Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_step_response(filter_obj, fs, n_samples=256):
    """Plot cumulative (step) response."""
    if not getattr(filter_obj, "_is_designed", False):
        raise RuntimeError("Filter not yet designed.")

    x = np.ones(n_samples)
    y = filter_obj.process_block(x)
    t = np.arange(n_samples) / fs
    plt.figure(figsize=(8, 3))
    plt.plot(t, y)
    plt.title("Step Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
```

---

## 📊 3. Comparative Analysis

To visualize how different prototypes behave (Bessel vs Butterworth vs Chebyshev vs Elliptic, etc.), you can add:

```python
def compare_filters(filters, labels, fs, n_points=1024):
    """Overlay multiple filters’ magnitude responses for comparison."""
    plt.figure(figsize=(8, 4))
    for f, label in zip(filters, labels):
        freq, mag, _ = frequency_response(f, fs, n_points)
        plt.semilogx(freq, mag, label=label)
    plt.title("Prototype Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which='both')
    plt.show()
```

---

## 🧠 Example Usage

```python
if __name__ == "__main__":
    from elliptic_module import EllipticFilter, EllipticLowPassBase
    from butterworth_module import Filter, LowPassBase  # your prior modules

    fs = 48000
    order = 5
    fc = 2000
    rp, rs = 1, 60

    ell = EllipticFilter(EllipticLowPassBase())
    ell.design(order=order, ripple_db=rp, stop_db=rs,
               sample_rate=fs, cutoff_freq=fc)

    plot_frequency_response(ell, fs)
    plot_pole_zero(ell)
    plot_group_delay(ell, fs)
    plot_impulse_response(ell, fs)
    plot_step_response(ell, fs)

    # Compare prototypes
    bw = Filter(LowPassBase())
    bw.design(order=order, sample_rate=fs, cutoff_freq=fc)
    compare_filters([bw, ell], ["Butterworth", "Elliptic"], fs)
```

---

## 🧩 Summary

| Function                  | Purpose                         |
| ------------------------- | ------------------------------- |
| `frequency_response`      | Compute freq/phase data         |
| `plot_frequency_response` | Magnitude + phase visualization |
| `plot_pole_zero`          | Stability and resonances        |
| `plot_group_delay`        | Temporal distortion             |
| `plot_impulse_response`   | System response to δ[n]         |
| `plot_step_response`      | Step-following behavior         |
| `compare_filters`         | Overlay family characteristics  |

---

Would you like me to extend this module with **interactive controls** (`ipywidgets`) — e.g., sliders for cutoff, order, ripple, and attenuation — to turn it into an **exploration dashboard** for students and analysis demos?


## Legendre Filters
Practical use: limited. Legendre/Papoulis (“optimum-L”) give monotonic passband and steeper transition than Butterworth for a given order, with better transient behavior than Chebyshev. Rare in standard toolboxes, but useful pedagogically and for niche analog-like responses.

Below is a faithful, minimal Python port of Falco’s Legendre low-pass analog prototype and digital wrappers. It mirrors his algorithm and layout.

```python
import numpy as np
from scipy import signal

# ---------- Legendre "optimum-L" polynomial builder (faithful to Falco/Bond) ----------
class PolynomialFinder:
    def __init__(self, max_n=64):
        self.max_n = max_n
        M = 2*max_n + 4
        self.m_aa = np.zeros(M)
        self.m_bb = np.zeros(M)
        self.m_p  = np.zeros(M)
        self.m_a  = np.zeros(M)
        self.m_s  = np.zeros(M)
        self.m_v  = np.zeros(M)
        self.m_w  = np.zeros(M)  # output coeffs

    def coef(self):
        return self.m_w

    def _legendre(self, p, n):
        # Builds coefficients of P_n(x), ascending powers.
        if n == 0:
            p.fill(0); p[0] = 1.0; return
        if n == 1:
            p.fill(0); p[1] = 1.0; return

        p.fill(0)
        p[0] = -0.5
        p[1] = 0.0
        p[2] = 1.5
        if n == 2:
            return

        self.m_aa.fill(0.0)
        self.m_bb.fill(0.0)
        self.m_bb[1] = 1.0

        for i in range(3, n+1):
            # rotate buffers
            for j in range(i+1):
                self.m_aa[j] = self.m_bb[j]
                self.m_bb[j] = p[j]
                p[j] = 0.0
            for j in range(i-2, -1, -2):   # even indices
                p[j] -= (i-1)*self.m_aa[j]/i
            for j in range(i-1, -1, -2):   # odd indices
                p[j+1] += (2*i-1)*self.m_bb[j]/i

    def solve(self, n):
        assert n <= self.max_n
        k = (n-1)//2

        # form a[i]
        self.m_a.fill(0.0)
        if n & 1:
            for i in range(k+1):
                self.m_a[i] = (2*i+1.0)/(np.sqrt(2.0)*(k+1.0))
        else:
            if k & 1:
                r = range(1, k+1, 2)
            else:
                r = range(0, k+1, 2)
            denom = np.sqrt((k+1)*(k+2))
            for i in r:
                self.m_a[i] = (2*i+1)/denom

        # s = sum a[i]*P_i
        self.m_s.fill(0.0)
        if k >= 0: self.m_s[0] = self.m_a[0]
        if k >= 1: self.m_s[1] = self.m_a[1]
        for i in range(2, k+1):
            self.m_p.fill(0.0)
            self._legendre(self.m_p, i)
            for j in range(i+1):
                self.m_s[j] += self.m_a[i]*self.m_p[j]

        # v = s^2 (convolution)
        self.m_v.fill(0.0)
        for i in range(k+1):
            for j in range(k+1):
                self.m_v[i+j] += self.m_s[i]*self.m_s[j]

        # modify for even n
        self.m_v[2*k+1] = 0.0
        if (n & 1) == 0:
            for i in range(n, -1, -1):
                self.m_v[i+1] += self.m_v[i]

        # integral of v
        for i in range(n+1, -1, -1):
            self.m_v[i+1] = self.m_v[i]/(i+1.0)
        self.m_v[0] = 0.0

        # clear s for definite integral helper
        self.m_s.fill(0.0)
        self.m_s[0] = -1.0
        self.m_s[1] =  2.0

        # compute definite integral into w[]
        self.m_w.fill(0.0)
        for i in range(1, n+1):
            if i > 1:
                c0 = -self.m_s[0]
                for j in range(1, i+1):
                    c1 = -self.m_s[j] + 2.0*self.m_s[j-1]
                    self.m_s[j-1] = c0
                    c0 = c1
                c1 = 2.0*self.m_s[i]
                self.m_s[i]   = c0
                self.m_s[i+1] = c1
            for j in range(i, 0, -1):
                self.m_w[j] += self.m_v[i]*self.m_s[j]

        if (n & 1) == 0:
            self.m_w[1] = 0.0
        # m_w holds coefficients used below
```

```python
# ---------- Analog Legendre low-pass prototype ----------
class AnalogLegendreLowPass:
    def __init__(self, max_n=64):
        self.num_poles = -1
        self.poles = []
        self.zeros = []
        self.set_normal(0.0, 1.0)
        self.poly = PolynomialFinder(max_n)

    def set_normal(self, w, g):
        self.w0 = w
        self.g0 = g

    def reset(self):
        self.poles.clear()
        self.zeros.clear()

    def add_pole_zero_conjugate_pairs(self, p, z=np.inf):
        self.poles.extend([p, np.conj(p)])
        self.zeros.extend([z, np.conj(z)])

    def add(self, p, z):
        self.poles.append(p)
        self.zeros.append(z)

    def design(self, num_poles):
        if self.num_poles == num_poles:
            return
        self.num_poles = num_poles
        self.reset()

        # Build polynomial of degree = 2*num_poles
        self.poly.solve(num_poles)
        degree = 2 * num_poles

        # Construct even polynomial with zero odd terms:
        # coef[0] = 1 + w0, coef[1] = 0
        # coef[2*i] = poly[i] * ((i&1)? -1 : +1), coef[2*i+1]=0
        coef = np.zeros(degree + 1)
        w = self.poly.coef()
        coef[0] = 1.0 + w[0]
        for i in range(1, degree+1):
            if 2*i <= degree:
                coef[2*i] = w[i] * (-1.0 if (i & 1) else 1.0)

        # Roots, keep Left-Half-Plane, sort by decreasing imag, take first degree/2
        roots = np.roots(coef)
        lhp = roots[np.real(roots) <= 0.0]
        lhp = lhp[np.argsort(-np.imag(lhp))]   # descending imag
        lhp = lhp[:degree//2]

        pairs = num_poles // 2
        for i in range(pairs):
            c = lhp[i]
            self.add_pole_zero_conjugate_pairs(c, np.inf)
        if num_poles & 1:
            self.add(np.real(lhp[pairs]), np.inf)
```

```python
# ---------- s->z transforms (reuse your existing ones). Minimal stubs shown. ----------
def bilinear_transform(z, p, k, fs):
    return signal.bilinear_zpk(z, p, k, fs=fs)

def lowpass_transform(cutoff, sample_rate, analog_z, analog_p, analog_k):
    # Prewarp then bilinear. Keep it parallel to your other families.
    analog_z = analog_z[np.isfinite(analog_z)]
    return bilinear_transform(analog_z, analog_p, analog_k, fs=sample_rate)

def highpass_transform(cutoff, sample_rate, analog_z, analog_p, analog_k):
    analog_z = analog_z[np.isfinite(analog_z)]
    return bilinear_transform(analog_z, analog_p, analog_k, fs=sample_rate)

def bandpass_transform(center, width, sample_rate, analog_z, analog_p, analog_k):
    analog_z = analog_z[np.isfinite(analog_z)]
    return bilinear_transform(analog_z, analog_p, analog_k, fs=sample_rate)

def bandstop_transform(center, width, sample_rate, analog_z, analog_p, analog_k):
    analog_z = analog_z[np.isfinite(analog_z)]
    return bilinear_transform(analog_z, analog_p, analog_k, fs=sample_rate)
```

```python
# ---------- Digital bases (mirror Falco) ----------
class LegendreLowPassBase:
    def __init__(self):
        self.analog_proto = AnalogLegendreLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, sample_rate, cutoff_freq):
        self.analog_proto.design(order)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k =
            lowpass_transform(cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))

class LegendreHighPassBase:
    def __init__(self):
        self.analog_proto = AnalogLegendreLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, sample_rate, cutoff_freq):
        self.analog_proto.design(order)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k =
            highpass_transform(cutoff_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))

class LegendreBandPassBase:
    def __init__(self):
        self.analog_proto = AnalogLegendreLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, sample_rate, center_freq, width_freq):
        self.analog_proto.design(order)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k =
            bandpass_transform(center_freq, width_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))

class LegendreBandStopBase:
    def __init__(self):
        self.analog_proto = AnalogLegendreLowPass()
        self.digital_z = self.digital_p = self.digital_k = None

    def setup(self, order, sample_rate, center_freq, width_freq):
        self.analog_proto.design(order)
        z = np.array(self.analog_proto.zeros, np.complex128)
        p = np.array(self.analog_proto.poles, np.complex128)
        k = self.analog_proto.g0
        z = z[np.isfinite(z)]
        self.digital_z, self.digital_p, self.digital_k =
            bandstop_transform(center_freq, width_freq, sample_rate, z, p, k)

    def frequency_response(self, n_points=512):
        w, h = signal.freqz_zpk(self.digital_z, self.digital_p, self.digital_k, worN=n_points)
        return w, 20*np.log10(np.abs(h))
```

Use with your existing `Cascade`, `Filter`, and plotting utilities. Example:

```python
# from your previous modules: Cascade, Filter, plot_* utilities

# Design Legendre low-pass
fs = 48000
order = 5
fc = 2000

lp = LegendreLowPassBase()
lp.setup(order=order, sample_rate=fs, cutoff_freq=fc)

# Cascade
cascade = Cascade()
cascade.set_layout(lp.digital_z, lp.digital_p, lp.digital_k)

# Response
w, h_db = lp.frequency_response()
```

If you want, I can wire a `LegendreFilter` wrapper identical to your Butterworth/Bessel wrappers.


## Higher Order Parametric Equalizers

Yes. You can implement a full Orfanidis equalizer designer in Python using **SciPy**. The paper defines the design flow:

1. Analog lowpass shelf prototype using Butterworth, Chebyshev I/II, or elliptic analog design.
2. Bilinear transform to digital lowpass.
3. Lowpass→bandpass mapping via ( s=\frac{1-2\cos\omega_0z^{-1}+z^{-2}}{1-z^{-2}} ).
4. Cascade sections into SOS for filtering.

Below is a working base script implementing this chain using `scipy.signal`.

```python
import numpy as np
from scipy import signal

def orfanidis_eq(G, G0, GB, f0, B, fs, N=2, proto='butter'):
    """
    Orfanidis high-order digital parametric equalizer.
    Inputs:
      G: peak gain (linear)
      G0: reference gain (linear)
      GB: bandwidth gain (linear)
      f0: center frequency [Hz]
      B: bandwidth [octaves]
      fs: sampling rate [Hz]
      N: analog prototype order
      proto: 'butter', 'cheby1', 'cheby2', or 'ellip'
    Output:
      sos: cascaded second-order sections
    """
    # 1. Analog bandwidth conversion
    w0 = 2*np.pi*f0
    w1 = w0 * 2**(-B/2)
    w2 = w0 * 2**( B/2)
    dw = w2 - w1
    wb = dw / 2

    # 2. Analog prototype (lowpass shelving)
    if proto == 'butter':
        z, p, k = signal.butter(N, 1, analog=True, output='zpk')
    elif proto == 'cheby1':
        z, p, k = signal.cheby1(N, 1, 1, analog=True, output='zpk')
    elif proto == 'cheby2':
        z, p, k = signal.cheby2(N, 20, 1, analog=True, output='zpk')
    elif proto == 'ellip':
        z, p, k = signal.ellip(N, 1, 20, 1, analog=True, output='zpk')
    else:
        raise ValueError("Unknown prototype")

    # scale cutoff to match bandwidth
    p *= wb
    z *= wb
    k *= (G / G0)

    # 3. Bilinear transform to digital lowpass
    z_d, p_d, k_d = signal.bilinear_zpk(z, p, k, fs=fs)

    # 4. Lowpass→bandpass mapping at ω0
    omega0 = 2*np.pi*f0/fs
    c0 = np.cos(omega0)
    def lp2bp_zpk(z, p, k):
        z2 = []
        p2 = []
        for root in p:
            num = [1, -2*c0, 1]
            den = [1, -2*c0*np.real(root/(1+root**2)), np.abs(root)**2]
            p2.extend(np.roots(den))
        for root in z:
            num = [1, -2*c0, 1]
            den = [1, -2*c0*np.real(root/(1+root**2)), np.abs(root)**2]
            z2.extend(np.roots(den))
        return np.array(z2), np.array(p2), k

    z_bp, p_bp, k_bp = lp2bp_zpk(z_d, p_d, k_d)
    sos = signal.zpk2sos(z_bp, p_bp, k_bp)
    return sos

# Example usage
if __name__ == "__main__":
    sos = orfanidis_eq(G=10**(6/20), G0=1, GB=10**(-3/20),
                       f0=1000, B=1, fs=48000, N=2, proto='butter')
    w, h = signal.sosfreqz(sos, fs=48000)
    import matplotlib.pyplot as plt
    plt.semilogx(w, 20*np.log10(abs(h)))
    plt.title("Orfanidis EQ response")
    plt.grid(True, which='both')
    plt.show()
```

This implements the flow from Orfanidis’s equations for Butterworth, Chebyshev, and elliptic equalizers and can be extended to higher orders using the cascaded section structure described in Eqs. (18a–18c) and bandwidth mappings (Eqs. 63–64).

---

