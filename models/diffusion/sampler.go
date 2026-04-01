package diffusion

import (
	"math"
	"math/rand"
)

// DiscreteFlowDenoiser implements the denoiser for Z-Image (Lumina2).
// shift=3.0, parameterization=v (velocity)
type DiscreteFlowDenoiser struct {
	Shift float32
}

// NewDiscreteFlowDenoiser creates a new denoiser with the given shift.
func NewDiscreteFlowDenoiser(shift float32) *DiscreteFlowDenoiser {
	return &DiscreteFlowDenoiser{Shift: shift}
}

// SigmaToT converts sigma to timestep value (0..1000 range).
// Matches sd.cpp: simply sigma * 1000 (NOT the mathematical inverse of TToSigma).
func (d *DiscreteFlowDenoiser) SigmaToT(sigma float32) float32 {
	return sigma * 1000.0
}

// TToSigma converts timestep (0..1000) to sigma.
func (d *DiscreteFlowDenoiser) TToSigma(t float32) float32 {
	tNorm := t / 1000.0
	return d.Shift * tNorm / (1.0 + (d.Shift-1.0)*tNorm)
}

// Denoise applies the denoiser scalings to get the predicted clean sample.
// For DiscreteFlow: x_denoised = (x - sigma * model_output) = x + sigma * model_output
// Actually: c_skip=1, c_out=-sigma, c_in=1
// x_denoised = c_skip * x + c_out * model_output = x - sigma * model_output
func (d *DiscreteFlowDenoiser) Denoise(modelOutput, x []float32, sigma float32) []float32 {
	n := len(x)
	denoised := make([]float32, n)
	for i := 0; i < n; i++ {
		denoised[i] = x[i] - sigma*modelOutput[i]
	}
	return denoised
}

// NoiseScaling prepares the noisy input from latent and noise.
// x_noisy = (1-sigma)*latent + sigma*noise
func (d *DiscreteFlowDenoiser) NoiseScaling(latent, noise []float32, sigma float32) []float32 {
	n := len(latent)
	noisy := make([]float32, n)
	for i := 0; i < n; i++ {
		noisy[i] = (1.0-sigma)*latent[i] + sigma*noise[i]
	}
	return noisy
}

// SimpleSchedule generates a simple linear sigma schedule.
// Returns sigmas array of length steps+1 (last element is 0).
func SimpleSchedule(steps int, denoiser *DiscreteFlowDenoiser) []float32 {
	sigmas := make([]float32, steps+1)
	for i := 0; i < steps; i++ {
		// Evenly spaced timesteps from 1000 down to near 0
		t := 1000.0 * float32(steps-i) / float32(steps)
		sigmas[i] = denoiser.TToSigma(t)
	}
	sigmas[steps] = 0.0
	return sigmas
}

// EulerSample runs the Euler ODE sampler for diffusion.
// model: function that runs the DiT model given (noisy_input, timestep) → model_output
// latentSize: number of elements in the latent
// steps: number of denoising steps
// seed: random seed for initial noise
// Returns the denoised latent.
func EulerSample(
	model func(x []float32, timestep float32) []float32,
	latentSize int,
	steps int,
	seed int64,
) []float32 {
	denoiser := NewDiscreteFlowDenoiser(3.0)
	sigmas := SimpleSchedule(steps, denoiser)

	// Generate initial noise
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, latentSize)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}

	// Scale initial noise by first sigma
	// Actually for DiscreteFlow, the initial x IS the noise scaled by the first sigma schedule
	// x_noisy = (1-sigma)*0 + sigma*noise = sigma*noise (since latent=0 initially)
	for i := range x {
		x[i] *= sigmas[0]
	}

	// Euler method
	for i := 0; i < steps; i++ {
		sigma := sigmas[i]
		sigmaNext := sigmas[i+1]

		// Convert sigma to timestep for the model
		// Z-Image (Lumina2) uses inverted timesteps: 1000 - t
		t := 1000.0 - denoiser.SigmaToT(sigma)

		// Run model
		modelOutput := model(x, t)

		// Get denoised prediction
		denoised := denoiser.Denoise(modelOutput, x, sigma)

		// Euler step: d = (x - denoised) / sigma
		// x_next = x + d * (sigma_next - sigma)
		dt := sigmaNext - sigma
		for j := range x {
			d := (x[j] - denoised[j]) / sigma
			x[j] += d * dt
		}

		_ = denoised // for GC
	}

	return x
}

// GaussianNoise generates standard normal noise.
func GaussianNoise(size int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	noise := make([]float32, size)
	for i := range noise {
		noise[i] = float32(rng.NormFloat64())
	}
	return noise
}

// BoxMullerNoise generates standard normal noise using Box-Muller transform,
// matching sd.cpp's random number generation for bit-exactness.
func BoxMullerNoise(size int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	noise := make([]float32, size)
	for i := 0; i < size; i += 2 {
		u1 := float64(rng.Float32())
		u2 := float64(rng.Float32())
		if u1 < 1e-7 {
			u1 = 1e-7
		}
		r := math.Sqrt(-2.0 * math.Log(u1))
		theta := 2.0 * math.Pi * u2
		noise[i] = float32(r * math.Cos(theta))
		if i+1 < size {
			noise[i+1] = float32(r * math.Sin(theta))
		}
	}
	return noise
}
