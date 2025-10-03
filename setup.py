from setuptools import setup

setup(
    name='slimevolleygym',
    version='0.1.0',
    keywords='games, environment, agent, rl, ai, gym',
    url='https://github.com/hardmaru/slimevolleygym',
    description='Slime Volleyball Gym Environment',
    packages=['slimevolleygym'],
    install_requires=[
        'gymnasium>=0.29.1',
        'numpy>=1.26.0',
        'opencv-python>=4.8.0',
        'pyglet>=1.5.0,<2.0.0'
    ]
)
