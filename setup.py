from setuptools import setup

setup(name='gym_auv',
      author='Narcis Palomeras',
      description='AUV Docking environment',
      url='git@github.com:narcispr/gym_auv.git',
      version='0.0.1',
      python_requires='>=3.6',
      install_requires=['gymnasium', 
                        'numpy',
                        'matplotlib',
                        'Pillow']
)

