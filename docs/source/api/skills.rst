.. _api_skills:

Agent skills
============

.. currentmodule:: climate_risk.skills

Agent skills ship inside the package, so any install carries them. ``install-climate-risk-skills``
places them where a coding agent looks, together with the user-guide pages they refer to.

Skills are written to ``.claude/skills/`` only when a ``.claude`` directory already exists, so
nothing is created for a harness that is not installed. A project install also keeps a block in the
project's ``AGENTS.md``, delimited by markers the installer owns.

What the package carries
------------------------

.. autosummary::
    :toctree: generated/

    ~bundle.available_skills
    ~bundle.docs_source
    ~agents_file.agents_block

Placing it on disk
------------------

.. autosummary::
    :toctree: generated/

    ~install.claude_skills_dir
    ~install.install_skill
    ~install.install_all
    ~agents_file.update_agents_file

Command line
------------

.. autosummary::
    :toctree: generated/

    ~cli.main
