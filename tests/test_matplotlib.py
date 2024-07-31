import matplotlib.pyplot as plt

# Update Matplotlib configuration to use LaTeX
plt.rcParams.update({
    "text.usetex": True,             # Use LaTeX for rendering text
    "font.family": "serif",          # Use serif fonts by default
    "font.serif": ["Computer Modern"],  # Use Computer Modern font
    "text.latex.preamble": r"\usepackage{amsmath}",  # Use AMS math package
})

# Create a plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
# Example LaTeX math in title
ax.set_title(r'$\frac{a}{b} \dot{\varepsilon} \times \sqrt{c}$', fontsize=20)
ax.set_xlabel(r'Time ($t$)', fontsize=16)
ax.set_ylabel(r'Value ($v$)', fontsize=16)

# Save the plot as a PDF file
plt.savefig("./test_matplotlib.pdf", format="pdf")

# Optionally, display the plot
plt.show()
