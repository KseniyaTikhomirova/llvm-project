import lit
import lit.formats
import platform

from lit.TestRunner import (
    ParserKind,
    IntegratedTestKeywordParser,
)

import os
import re

class SYCLEndToEndTest(lit.formats.ShTest):
    def parseTestScript(self, test):
        """This is based on lit.TestRunner.parseIntegratedTestScript but we
        overload the semantics of REQUIRES/UNSUPPORTED/XFAIL directives so have
        to implement it manually."""

        # Parse the test sources and extract test properties
        try:
            parsed = lit.TestRunner._parseKeywords(
                test.getSourcePath(),
                additional_parsers=[
                # will be added later
                ],
                require_script=True,
            )
        except ValueError as e:
            return lit.Test.Result(lit.Test.UNRESOLVED, str(e))
        script = parsed["RUN:"] or []
        assert parsed["DEFINE:"] == script
        assert parsed["REDEFINE:"] == script

        test.xfails += parsed["XFAIL:"] or []
        test.requires += test.config.required_features
        test.requires += parsed["REQUIRES:"] or []
        test.unsupported += test.config.unsupported_features
        test.unsupported += parsed["UNSUPPORTED:"] or []

        return script

    def execute(self, test, litConfig):
        if test.config.unsupported:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Test is unsupported")

        filename = test.path_in_suite[-1]
        tmpDir, tmpBase = lit.TestRunner.getTempPaths(test)
        script = self.parseTestScript(test)
        if isinstance(script, lit.Test.Result):
            return script

        substitutions = lit.TestRunner.getDefaultSubstitutions(test, tmpDir, tmpBase)

        substitutions.append(
            (
                "%{build}",
                "%clangxx -fsycl %verbose_print %s",
            )
        )

        new_script = []
        for directive in script:
            if not isinstance(directive, lit.TestRunner.CommandDirective):
                new_script.append(directive)
                continue

            if "%{run}" not in directive.command:
                new_script.append(directive)
                continue

        script = new_script

        conditions = {feature: True for feature in test.config.available_features}
        script = lit.TestRunner.applySubstitutions(
            script,
            substitutions,
            conditions,
            recursion_limit=test.config.recursiveExpansionLimit,
        )

        if len(script) == 0:
            return lit.Test.Result(lit.Test.UNSUPPORTED, "Lit script is empty")

        result = lit.TestRunner._runShTest(test, litConfig, False, script, tmpBase)

        # Single triple/device - might be an XFAIL.
        def map_result(features, code):
            if "*" in test.xfails:
                if code is lit.Test.PASS:
                    code = lit.Test.XPASS
                elif code is lit.Test.FAIL:
                    code = lit.Test.XFAIL
            return code

        # Set this to empty so internal lit code won't change our result if it incorrectly
        # thinks the test should XFAIL. This can happen when our XFAIL condition relies on
        # device features, since the internal lit code doesn't have knowledge of these.
        test.xfails = []

        return result
