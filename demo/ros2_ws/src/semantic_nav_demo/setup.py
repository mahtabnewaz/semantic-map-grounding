import os
from glob import glob
from setuptools import setup

package_name = "semantic_nav_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="demo",
    maintainer_email="you@example.com",
    description="Language-grounded navigation demo bridge.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "agent_bridge = semantic_nav_demo.agent_bridge:main",
        ],
    },
)
