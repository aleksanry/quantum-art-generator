import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from qiskit import QuantumCircuit, Aer, execute
from skimage.transform import rotate
import random
from collections import deque
import hashlib

class QuantumArtGenerator:
    def __init__(self, canvas_size=512):
        self.canvas_size = canvas_size
        self.backend = Aer.get_backend('statevector_simulator')
        self.population = deque(maxlen=20)  # Stores previous artworks
        self.initialize_quantum_palette()
        
    def initialize_quantum_palette(self):
        """Creates a color palette using quantum superposition"""
        qc = QuantumCircuit(3)
        qc.h(0)  # Put first qubit in superposition
        qc.cx(0, 1)  # Entangle with second qubit
        qc.ry(np.pi/3, 2)  # Rot third qubit
        
        result = execute(qc, self.backend).result()
        statevector = result.get_statevector()
        
        self.colors = []
        for i, amplitude in enumerate(statevector):
            prob = abs(amplitude)**2
            hue = i/8  # 3 qubits = 8 states
            saturation = 0.7 + 0.3 * prob
            value = 0.5 + 0.5 * prob
            self.colors.append((hue, saturation, value))
    
    def quantum_measurement(self, num_qubits=3):
        """Simulates quantum measurement to make artistic decisions"""
        qc = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            qc.ry(random.uniform(0, 2*np.pi), qubit)
        result = execute(qc, self.backend).result()
        statevector = result.get_statevector()
        probabilities = [abs(amp)**2 for amp in statevector]
        return random.choices(range(2**num_qubits), weights=probabilities)[0]
    
    def generate_artwork(self, user_preference=None):
        """Creates a new artwork using quantum-inspired rules"""
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3))
        
        # Quantum-inspired parameters
        num_shapes = self.quantum_measurement(2) + 1  # 1-4 shapes
        symmetry_type = self.quantum_measurement(1)  # Symmetric or asymmetric
        
        for _ in range(num_shapes):
            # Quantum color selection
            color_idx = self.quantum_measurement(3)
            h, s, v = self.colors[color_idx]
            
            # Quantum position and size
            x = self.quantum_measurement(4) / 16
            y = self.quantum_measurement(4) / 16
            size = 0.1 + 0.3 * (self.quantum_measurement(2) / 4)
            
            # Create shape with quantum characteristics
            shape = self.create_quantum_shape(h, s, v, x, y, size)
            
            # Apply symmetry if selected
            if symmetry_type:
                shape = self.apply_symmetry(shape)
                
            # Blend with canvas
            canvas = np.maximum(canvas, shape)
        
        # Apply quantum-inspired post-processing
        canvas = self.quantum_filter(canvas)
        
        # Store in population
        art_hash = hashlib.md5(canvas.tobytes()).hexdigest()
        self.population.append((art_hash, canvas))
        
        return canvas
    
    def create_quantum_shape(self, h, s, v, x, y, size):
        """Creates a shape with quantum-probabilistic edges"""
        xx, yy = np.mgrid[0:self.canvas_size, 0:self.canvas_size]
        xx = xx / self.canvas_size
        yy = yy / self.canvas_size
        
        # Quantum probability distribution for shape
        dist = np.sqrt((xx - x)**2 + (yy - y)**2) / size
        q_prob = np.cos(dist * np.pi * 2)**2  # Born rule probability
        
        # Add quantum noise
        noise = np.random.random((self.canvas_size, self.canvas_size)) * 0.3
        q_prob = np.clip(q_prob + noise, 0, 1)
        
        # Create HSV then convert to RGB
        h_channel = np.ones_like(q_prob) * h
        s_channel = np.ones_like(q_prob) * s
        v_channel = q_prob * v
        hsv = np.dstack((h_channel, s_channel, v_channel))
        rgb = hsv_to_rgb(hsv)
        
        return rgb
    
    def apply_symmetry(self, shape):
        """Applies quantum-inspired symmetry transformations"""
        symmetry = self.quantum_measurement(2)
        if symmetry == 0:  # Radial
            return np.maximum(shape, rotate(shape, 90, reshape=False))
        elif symmetry == 1:  # Horizontal
            return np.maximum(shape, np.fliplr(shape))
        else:  # Vertical
            return np.maximum(shape, np.flipud(shape))
    
    def quantum_filter(self, canvas):
        """Applies quantum-style post-processing"""
        # Create entanglement-inspired effect
        entanglement = np.fft.fft2(canvas[:,:,0])
        entanglement = np.fft.fftshift(entanglement)
        mask = np.zeros_like(entanglement)
        center = self.canvas_size//2
        mask[center-30:center+30, center-30:center+30] = 1
        entanglement *= mask
        entanglement = np.fft.ifftshift(entanglement)
        entanglement = np.fft.ifft2(entanglement).real
        
        # Superimpose entanglement effect
        canvas[:,:,0] = np.clip(canvas[:,:,0] + entanglement*0.1, 0, 1)
        return canvas
    
    def evolve_artwork(self, preference_vector):
        """Evolves the artwork based on user preferences"""
        if not self.population:
            return self.generate_artwork()
            
        # Select parents based on preference
        parent_idx = min(len(self.population)-1, int(preference_vector[0] * len(self.population)))
        parent = self.population[parent_idx][1]
        
        # Create child with quantum variations
        child = parent.copy()
        
        # Apply quantum mutations
        mutation_type = self.quantum_measurement(2)
        if mutation_type == 0:  # Color shift
            shift = (preference_vector[1] - 0.5) * 0.3
            child = np.roll(child, int(shift * len(self.colors)), axis=2)
        elif mutation_type == 1:  # Shape distortion
            amount = preference_vector[2] * 0.2
            child = np.fft.fft2(child, axes=(0,1))
            phase = np.exp(1j * amount * np.random.randn(*child.shape[:2]))
            for i in range(3):
                child[:,:,i] *= phase
            child = np.fft.ifft2(child, axes=(0,1)).real
        else:  # Pattern replication
            factor = int(1 + preference_vector[3] * 3)
            child = np.tile(child, (factor, factor, 1))[:self.canvas_size, :self.canvas_size]
        
        return np.clip(child, 0, 1)

# Example usage
if __name__ == "__main__":
    generator = QuantumArtGenerator()
    
    # Generate initial artwork
    artwork = generator.generate_artwork()
    plt.imshow(artwork)
    plt.axis('off')
    plt.show()
    
    # Evolve based on user preferences (simulated)
    for i in range(5):
        print(f"Generating evolved artwork {i+1}")
        preferences = np.random.rand(4)  # Simulated user preferences
        artwork = generator.evolve_artwork(preferences)
        plt.imshow(artwork)
        plt.axis('off')
        plt.show()