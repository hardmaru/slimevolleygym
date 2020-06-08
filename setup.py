from setuptools import setup

setup(
    name='slimevolleygym',
    version='0.1.0',
    keywords='games, environment, agent, rl, ai, gym',
    url='https://github.com/hardmaru/SlimeVolleyGym',
    description='Slime Volleyball Gym Environment',
    packages=['slimevolleygym'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
