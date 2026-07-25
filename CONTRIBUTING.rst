Contributing to ParMOO
======================

Contributions of source code, documentations, and fixes are happily
accepted via GitHub pull request to

    https://github.com/parmoo/parmoo/tree/develop

If you are planning a contribution, reporting bugs, or suggesting features, 
we encourage you to discuss the concept by opening a github issue at

  https://github.com/parmoo/parmoo/issues
  
or by emailing  ``parmoo@lbl.gov``
and interacting with us to ensure that your effort is well-directed.

Contribution Process
--------------------

ParMOO uses the Gitflow model. Contributors should typically branch from, and
make pull requests to, the ``develop`` branch. The ``main`` branch is used only
for releases. Pull requests may be made from a fork, for those without
repository write access.

Issues can be raised at

    https://github.com/parmoo/parmoo/issues

Issues may include reporting bugs or suggested features.

By convention, user branch names should have a ``<type>/<name>`` format, where
example types are ``feature``, ``bugfix``, ``testing``, ``docs``, and
``experimental``.
Administrators may take a ``hotfix`` branch from the main, which will be
merged into ``main`` (as a patch) and ``develop``.
Administrators may also take a ``release`` branch off ``develop`` and then
merge this branch into ``main`` and ``develop`` for a release.

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number.

New features should be accompanied by at least one test case.

All pull requests to ``develop`` or ``main`` must be reviewed by at least one
administrator.


Guidelines for AI/LLM-Assisted Contributions
--------------------

- **Remain accountable for all your outputs and decisions.**
   Individuals remain fully responsible and accountable for the accuracy, quality, appropriateness, and consequences of their work. Use of AI does not transfer this responsibility to the AI model, agent, or other tool.
- **Understand your work.**
   Regardless of how code or a pull request was produced, this project requires that authors illustrate a thorough understanding of any proposed changes. You must review such code line-by-line; it is your responsibility to ensure that it is correct, and that it does not breach copyright. Always critically engage with AI outputs, do not trust them implicitly. AI-assisted code, analysis, and artifacts must be tested and validated at a level appropriate to their impact. Authors are responsible for ensuring that generated code is correct, secure, maintainable, non-obfuscated, appropriately scoped, documented, and reproducible where relevant.
- **Disclose AI-generated or AI-assisted work.**
   If AI/LLM tools were primarily used to generate code or artifacts, this should be clearly indicated in the pull request. 
- **Use of AI to review pull requests.**
   All pull requests must be reviewed by a human reviewer. An LLM review may be used in addition to a human reviewer since this can help spot issues that a human may have missed, but this should not be the sole reviewer. The human reviewer should be fully accountable and responsible for the review feedback or comments (see 1).
- **Proprietary or personal information.**
   For this project, proprietary or personal information should never be sent to code generators or AI tools.
- **Be transparent, assume goodwill, and share what you learn.**
   Contributors should be open about relevant AI use, disclose details of AI use as appropriate to the project, engage constructively with colleagues, and share experiences and lessons learned with the project.

Developer's Certificate
-----------------------

ParMOO is distributed under a 3-clause BSD license (see LICENSE_).  
The act of submitting a pull request or patch will be understood as an 
affirmation that the developer acknowledges the above
`Guidelines for AI/LLM-Assisted Contributions 
<CONTRIBUTING.rst#guidelines-for-aillm-assisted-contributions>`_
and affirms the following:

::

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.


.. _LICENSE: https://github.com/parmoo/parmoo/blob/main/LICENSE
