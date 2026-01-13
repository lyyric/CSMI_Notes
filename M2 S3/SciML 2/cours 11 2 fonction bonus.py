def _calculate_conv_sizes(self, input_size, kernel_size, padding):
"""Calcule les tailles après chaque convolution"""
def conv_output_size(input_size, kernel_size, stride, padding):
return (input_size + 2 * padding - kernel_size) // stride + 1
 
size1 = conv_output_size(input_size, kernel_size, 2, padding)
size2 = conv_output_size(size1, kernel_size, 2, padding)
size3 = conv_output_size(size2, kernel_size, 2, padding)
 
return (size1, size2, size3)
 
def _calculate_output_paddings(self, kernel_size, padding):
"""Calcule les output_padding nécessaires pour reconstruire la taille exacte"""
sizes = self.sizes_after_conv
 
# Pour revenir à la taille précédente avec ConvTranspose
def needed_output_padding(target_size, input_size, kernel_size, stride, padding):
# Formule: output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
expected = (input_size - 1) * stride - 2 * padding + kernel_size
return max(0, target_size - expected)
 
# De sizes[2] vers sizes[1]
out_pad1 = needed_output_padding(sizes[1], sizes[2], kernel_size, 2, padding)
# De sizes[1] vers sizes[0]
out_pad2 = needed_output_padding(sizes[0], sizes[1], kernel_size, 2, padding)
# De sizes[0] vers input_dim
out_pad3 = needed_output_padding(self.input_dim, sizes[0], kernel_size, 2, padding)
 
return (out_pad1, out_pad2, out_pad3)