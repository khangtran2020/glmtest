import os
import sys
import git
import json
from typing import List
from transformers import PreTrainedModel, PreTrainedTokenizer
from rich.console import Console
from data.core import Data
from graph.core import Graph
from git import Repo
import importlib.util
import inspect as insp

REPO_MODULE_DICT = {
    "ansible": [
        "ansible.plugins.lookup.sequence",
        "ansible.module_utils.facts.system.fips",
        "ansible.executor.discovery.python_target",
        "ansible.cli.adhoc",
        "ansible.module_utils.facts.virtual.sysctl",
        "ansible.module_utils.facts.network.darwin",
        "ansible.playbook.taggable",
        "ansible.modules.iptables",
        "ansible.constants",
        "ansible.plugins.filter.urls",
        "ansible.module_utils.common.json",
        "ansible.module_utils.facts.utils",
        "ansible.modules.replace",
        "ansible.utils.lock",
        "ansible.module_utils.common.arg_spec",
        "ansible.plugins.strategy.debug",
        "ansible.module_utils.common.locale",
        "ansible.module_utils.facts.system.apparmor",
        "ansible.module_utils.pycompat24",
        "ansible.module_utils.common.parameters",
        "ansible.plugins.become.su",
        "ansible.module_utils.facts.network.aix",
        "ansible.utils.unsafe_proxy",
        "ansible.plugins.lookup.file",
        "ansible.module_utils.common.network",
        "ansible.plugins.shell.powershell",
        "ansible.module_utils.facts.virtual.hpux",
        "ansible.playbook.role.definition",
        "ansible.plugins.action.debug",
        "ansible.parsing.utils.yaml",
        "ansible.module_utils.connection",
        "ansible.template.vars",
        "ansible.plugins.action.assert",
        "ansible.module_utils.common.validation",
        "ansible.module_utils.facts.ansible_collector",
        "ansible.parsing.utils.jsonify",
        "ansible.modules.debconf",
        "ansible.module_utils.yumdnf",
        "ansible.plugins.become.sudo",
        "ansible.utils.jsonrpc",
        "ansible.parsing.yaml.loader",
        "ansible.modules.yum_repository",
        "ansible.executor.playbook_executor",
        "ansible.plugins.inventory.ini",
        "ansible.module_utils.facts.virtual.linux",
        "ansible.executor.process.worker",
        "ansible.playbook.playbook_include",
        "ansible.plugins.action.normal",
        "ansible.utils.helpers",
        "ansible.plugins.loader",
        "ansible.modules.command",
        "ansible.vars.plugins",
        "ansible.module_utils.facts.system.local",
        "ansible.module_utils.facts.hardware.openbsd",
        "ansible.module_utils.facts.system.caps",
        "ansible.playbook.helpers",
        "ansible.module_utils.facts.system.selinux",
        "ansible.module_utils.facts.virtual.sunos",
        "ansible.plugins.action.include_vars",
        "ansible.module_utils.facts.hardware.darwin",
        "ansible.cli.inventory",
        "ansible.module_utils.common.process",
        "ansible.playbook.play_context",
        "ansible.module_utils.facts.system.chroot",
        "ansible.cli.vault",
        "ansible.plugins.inventory.auto",
        "ansible.template.native_helpers",
        "ansible.module_utils.facts.network.fc_wwn",
        "ansible.modules.getent",
        "ansible.module_utils.facts.network.hurd",
        "ansible.plugins.action.set_stats",
        "ansible.module_utils.facts.system.platform",
        "ansible.playbook.task",
        "ansible.plugins.cache.jsonfile",
        "ansible.vars.clean",
        "ansible.template.safe_eval",
        "ansible.plugins.lookup.unvault",
        "ansible.playbook.block",
        "ansible.module_utils.facts.virtual.openbsd",
        "ansible.utils.py3compat",
        "ansible.modules.pip",
        "ansible.plugins.action.shell",
        "ansible.module_utils.common.dict_transformations",
        "ansible.module_utils.compat.selinux",
        "ansible.plugins.inventory.host_list",
        "ansible.playbook.handler_task_include",
        "ansible.cli.scripts.ansible_connection_cli_stub",
        "ansible.module_utils.facts.hardware.hurd",
        "ansible.plugins.callback.tree",
        "ansible.utils.singleton",
        "ansible.playbook.task_include",
        "ansible.module_utils.facts.system.python",
        "ansible.parsing.quoting",
        "ansible.module_utils.facts.sysctl",
        "ansible.inventory.data",
        "ansible.plugins.inventory.generator",
        "ansible.module_utils.facts.system.ssh_pub_keys",
        "ansible.playbook.collectionsearch",
        "ansible.cli.playbook",
        "ansible.plugins.filter.core",
        "ansible.cli.arguments.option_helpers",
        "ansible.module_utils.facts.hardware.netbsd",
        "ansible.module_utils.compat.version",
        "ansible.utils.version",
        "ansible.plugins.connection.psrp",
        "ansible.plugins.action.unarchive",
        "ansible.utils.color",
        "ansible.plugins.lookup.url",
        "ansible.playbook.role.metadata",
        "ansible.module_utils.facts.other.ohai",
        "ansible.plugins.lookup.subelements",
        "ansible.modules.systemd",
        "ansible.plugins.lookup.random_choice",
        "ansible.plugins.action.wait_for_connection",
        "ansible.galaxy.api",
        "ansible.parsing.yaml.objects",
        "ansible.playbook.included_file",
        "ansible.context",
        "ansible.plugins.lookup.nested",
        "ansible.plugins.action.yum",
        "ansible.module_utils.facts.system.service_mgr",
        "ansible.module_utils.facts.hardware.hpux",
        "ansible.modules.dnf",
        "ansible.plugins.callback.oneline",
        "ansible.plugins.lookup.template",
        "ansible.plugins.filter.mathstuff",
        "ansible.plugins.inventory.toml",
        "ansible.plugins.inventory.advanced_host_list",
        "ansible.parsing.yaml.dumper",
        "ansible.plugins.lookup.ini",
        "ansible.inventory.group",
        "ansible.plugins.strategy.host_pinned",
        "ansible.plugins.strategy.free",
        "ansible.modules.expect",
        "ansible.module_utils.facts.virtual.freebsd",
        "ansible.plugins.lookup.csvfile",
        "ansible.vars.hostvars",
        "ansible.playbook.base",
        "ansible.executor.interpreter_discovery",
        "ansible.inventory.helpers",
        "ansible.playbook.attribute",
        "ansible.plugins.connection.paramiko_ssh",
        "ansible.module_utils.facts.system.user",
        "ansible.module_utils.facts.network.hpux",
        "ansible.plugins.action.validate_argument_spec",
        "ansible.playbook.role.requirement",
        "ansible.utils.vars",
        "ansible.utils._junit_xml",
        "ansible.utils.collection_loader._collection_finder",
        "ansible.plugins.filter.urlsplit",
        "ansible.plugins.action.reboot",
        "ansible.executor.task_result",
        "ansible.module_utils.facts.other.facter",
        "ansible.module_utils.facts.virtual.netbsd",
        "ansible.module_utils.facts.system.distribution",
        "ansible.utils.context_objects",
        "ansible.module_utils.facts.system.dns",
        "ansible.modules.apt_repository",
        "ansible.module_utils.splitter",
        "ansible.plugins.lookup.first_found",
        "ansible.parsing.yaml.constructor",
        "ansible.cli.doc",
        "ansible.config.data",
        "ansible.module_utils.facts.compat",
        "ansible.plugins.callback.junit",
        "ansible.playbook.conditional",
        "ansible.plugins.action.fail",
        "ansible.plugins.action.assemble",
        "ansible.module_utils.facts.packages",
        "ansible.plugins.lookup.vars",
        "ansible.executor.play_iterator",
        "ansible.plugins.action.fetch",
        "ansible.utils.listify",
        "ansible.utils.collection_loader._collection_config",
        "ansible.module_utils.common.collections",
        "ansible.module_utils.facts.system.lsb",
        "ansible.executor.powershell.module_manifest",
        "ansible.playbook.role_include",
        "ansible.playbook.role.include",
        "ansible.modules.sysvinit",
        "ansible.plugins.strategy.linear",
        "ansible.plugins.inventory.constructed",
        "ansible.playbook.play",
        "ansible.module_utils.facts.system.date_time",
        "ansible.plugins.action.copy",
        "ansible.modules.subversion",
        "ansible.plugins.lookup.varnames",
        "ansible.plugins.action.set_fact",
        "ansible.parsing.splitter",
        "ansible.modules.cron",
        "ansible.parsing.mod_args",
        "ansible.module_utils.facts.system.cmdline",
        "ansible.module_utils.common._utils",
        "ansible.plugins.filter.encryption",
        "ansible.plugins.action.service",
        "ansible.plugins.lookup.config",
        "ansible.cli.console",
        "ansible.module_utils.facts.hardware.aix",
        "ansible.vars.manager",
        "ansible.utils.hashing",
        "ansible.modules.ping",
        "ansible.module_utils.common.text.formatters",
        "ansible.config.manager",
        "ansible.module_utils.facts.hardware.sunos",
        "ansible.inventory.manager",
        "ansible.modules.rpm_key",
        "ansible.module_utils.facts.network.generic_bsd",
        "ansible.plugins.action.gather_facts",
        "ansible.executor.stats",
        "ansible.module_utils.urls",
        "ansible.parsing.ajson",
        "ansible.module_utils.common.text.converters",
        "ansible.plugins.lookup.together",
        "ansible.module_utils.api",
        "ansible.vars.reserved",
        "ansible.plugins.action.group_by",
        "ansible.plugins.vars.host_group_vars",
        "ansible.modules.dpkg_selections",
        "ansible.module_utils.facts.network.sunos",
        "ansible.collections.list",
        "ansible.module_utils.facts.collector",
        "ansible.parsing.utils.addresses",
        "ansible.module_utils.common.sys_info",
        "ansible.plugins.action.pause",
        "ansible.module_utils.facts.hardware.freebsd",
        "ansible.plugins.callback.minimal",
        "ansible.modules.slurp",
        "ansible.plugins.inventory.yaml",
        "ansible.plugins.lookup.fileglob",
        "ansible.plugins.lookup.inventory_hostnames",
        "ansible.galaxy.token",
        "ansible.parsing.dataloader",
        "ansible.vars.fact_cache",
        "ansible.plugins.callback.default",
        "ansible.modules.lineinfile",
        "ansible.module_utils.facts.network.linux",
        "ansible.inventory.host",
    ],
    "cookiecutter": [
        "cookiecutter.repository",
        "cookiecutterfile",
        "cookiecutter.prompt",
        "cookiecutter.replay",
        "cookiecutter.find",
    ],
    "thef": [
        "thef.rules.brew_install",
        "thef.shells.generic",
        "thef.rules.git_rm_recursive",
        "thef.rules.pacman_invalid_option",
        "thef.rules.cp_create_destination",
        "thef.rules.choco_install",
        "thef.rules.vagrant_up",
        "thef.entrypoints.fix_command",
        "thef.rules.aws_cli",
        "thef.rules.no_such_file",
        "thef.logs",
        "thef.conf",
        "thef.entrypoints.not_configured",
        "thef.argument_parser",
        "thef.rules.git_push_pull",
        "thef.rules.tsuru_not_command",
        "thef.entrypoints.shell_logger",
        "thef.rules.django_south_merge",
        "thef.entrypoints.alias",
        "thef.rules.lein_not_task",
        "thef.rules.git_diff_no_index",
        "thef.rules.rm_root",
        "thef.entrypoints.main",
        "thef.rules.scm_correction",
        "thef.rules.sudo_command_from_user_path",
        "thef.system.unix",
        "thef.corrector",
        "thef.rules.dirty_unzip",
        "thef.rules.git_add_force",
        "thef.rules.cat_dir",
        "thef.types",
        "thef.rules.git_commit_reset",
    ],
    "youtube_dl": [
        "youtube_dl.extractor.archiveorg",
        "youtube_dl.extractor.eitb",
        "youtube_dl.extractor.safari",
        "youtube_dl.postprocessor.xattrpp",
        "youtube_dl.extractor.nrk",
        "youtube_dl.extractor.walla",
        "youtube_dl.downloader.common",
        "youtube_dl.extractor.soundgasm",
        "youtube_dl.aes",
        "youtube_dl.extractor.fourtube",
        "youtube_dl.postprocessor.metadatafromtitle",
        "youtube_dl.extractor.heise",
        "youtube_dl.downloader.f4m",
        "youtube_dl.downloader.ism",
        "youtube_dl.swfinterp",
        "youtube_dl.extractor.tudou",
        "youtube_dl.downloader.hls",
        "youtube_dl.extractor.tf1",
        "youtube_dl.extractor.konserthusetplay",
        "youtube_dl.downloader.http",
        "youtube_dl.jsinterp",
        "youtube_dl.extractor.thestar",
        "youtube_dl.socks",
        "youtube_dl.postprocessor.common",
        "youtube_dl.extractor.trutv",
        "youtube_dl.extractor.glide",
        "youtube_dl.extractor.zdf",
        "youtube_dl.extractor.udn",
        "youtube_dl.downloader.fragment",
        "youtube_dl.extractor.linuxacademy",
        "youtube_dl.extractor.tvplay",
        "youtube_dl.downloader.dash",
        "youtube_dl.extractor.hitrecord",
        "youtube_dl.options",
        "youtube_dl.extractor.itv",
    ],
    "mimesis": [
        "mimesis.providers.payment",
        "mimesis.providers.person",
        "mimesis.random",
        "mimesis.providers.address",
        "mimesis.decorators",
        "mimesis.schema",
        "mimesis.providers.path",
        "mimesis.providers.structure",
        "mimesis.builtins.pl",
        "mimesis.builtins.pt_br",
        "mimesis.builtins.en",
        "mimesis.builtins.ru",
        "mimesis.providers.base",
        "mimesis.providers.internet",
        "mimesis.providers.generic",
        "mimesis.providers.choice",
        "mimesis.providers.text",
        "mimesis.providers.cryptographic",
    ],
    "tqdm": [
        "tqdm.auto",
        "tqdm.contrib.utils_worker",
        "tqdm.rich",
        "tqdm._tqdm_pandas",
        "tqdm.contrib.logging",
        "tqdm.notebook",
        "tqdm.gui",
        "tqdm.contrib.telegram",
        "tqdm.contrib.itertools",
    ],
    "sanic": [
        "sanic.blueprint_group",
        "sanic.helpers",
        "sanic.mixins.middleware",
        "sanic.mixins.routes",
        "sanic.mixins.exceptions",
        "sanic.headers",
        "sanic.cookies",
        "sanic.utils",
        "sanic.exceptions",
        "sanic.router",
        "sanic.response",
    ],
    "httpie": [
        "httpie.client",
        "httpie.context",
        "httpie.output.streams",
        "httpie.plugins.base",
        "httpie.core",
        "httpie.utils",
        "httpie.cli.requestitems",
        "httpie.output.formatters.headers",
        "httpie.output.formatters.colors",
        "httpie.output.processing",
        "httpie.output.formatters.json",
        "httpie.config",
        "httpie.uploads",
        "httpie.cli.argparser",
        "httpie.sessions",
        "httpie.models",
        "httpie.cli.definition",
        "httpie.output.writer",
        "httpie.plugins.manager",
    ],
    "pysnooper": [
        "pysnooper.tracer",
        "pysnooper.pycompat",
        "pysnooper.variables",
        "pysnooper.utils",
    ],
    "pytutils": [
        "pytutils.urls",
        "pytutils.excs",
        "pytutils.env",
        "pytutils.files",
        "pytutils.lazy.lazy_regex",
        "pytutils.lazy.simple_import",
        "pytutils.python",
        "pytutils.props",
        "pytutils.trees",
        "pytutils.lazy.lazy_import",
        "pytutils.path",
        "pytutils.log",
    ],
    "typesystem": [
        "typesystem.schemas",
        "typesystem.tokenize.tokenize_json",
        "typesystem.base",
        "typesystem.tokenize.positional_validation",
        "typesystem.formats",
        "typesystem.json_schema",
        "typesystem.tokenize.tokenize_yaml",
        "typesystem.fields",
        "typesystem.composites",
        "typesystem.tokenize.tokens",
    ],
    "codetiming": ["codetiming._timers"],
    "dataclasses_json": [
        "dataclasses_json.mm",
        "dataclasses_json.core",
        "dataclasses_json.undefined",
        "dataclasses_json.cfg",
    ],
    "tornado": [
        "tornado.locks",
        "tornado.options",
        "tornado.escape",
        "tornado.locale",
        "tornado.httpclient",
        "tornado.auth",
        "tornado.simple_httpclient",
        "tornado.queues",
        "tornado.concurrent",
        "tornado.netutil",
        "tornado.util",
        "tornado.log",
        "tornado.tcpclient",
    ],
    "flutils": [
        "flutils.setuputils.cfg",
        "flutils.packages",
        "flutils.objutils",
        "flutils.codecs.b64",
        "flutils.namedtupleutils",
        "flutils.codecs.raw_utf8_escape",
        "flutils.txtutils",
        "flutils.pathutils",
        "flutils.decorators",
    ],
    "pymonet": [
        "pymonet.task",
        "pymonet.validation",
        "pymonet.immutable_list",
        "pymonet.either",
        "pymonet.box",
        "pymonet.maybe",
        "pymonet.utils",
        "pymonet.monad_try",
        "pymonet.lazy",
        "pymonet.semigroups",
    ],
    "py_backwards": [
        "py_backwards.transformers.string_types",
        "py_backwards.transformers.base",
        "py_backwards.types",
        "py_backwards.transformers.dict_unpacking",
        "py_backwards.transformers.yield_from",
        "py_backwards.transformers.starred_unpacking",
        "py_backwards.transformers.return_from_generator",
        "py_backwards.transformers.variables_annotations",
        "py_backwards.utils.helpers",
        "py_backwards.conf",
        "py_backwards.utils.snippet",
        "py_backwards.transformers.metaclass",
        "py_backwards.transformers.six_moves",
        "py_backwards.utils.tree",
        "py_backwards.transformers.python2_future",
        "py_backwards.files",
        "py_backwards.transformers.super_without_arguments",
        "py_backwards.main",
        "py_backwards.compiler",
    ],
    "flutes": ["flutes.timing", "flutes.iterator", "flutes.structure"],
    "pypara": [
        "pypara.commons.errors",
        "pypara.monetary",
        "pypara.accounting.ledger",
        "pypara.exchange",
        "pypara.accounting.journaling",
        "pypara.dcc",
    ],
    "semantic_release": [
        "semantic_release.settings",
        "semantic_release.helpers",
        "semantic_release.hvcs",
        "semantic_release.pypi",
        "semantic_release.ci_checks",
        "semantic_release.dist",
    ],
    "thonny": [
        "thonny.jedi_utils",
        "thonny.roughparse",
        "thonny.plugins.pgzero_frontend",
    ],
    "string_utils": [
        "string_utils.validation",
        "string_utils.manipulation",
        "string_utils.generation",
    ],
    "docstring_parser": [
        "docstring_parser.rest",
        "docstring_parser.parser",
        "docstring_parser.common",
        "docstring_parser.numpydoc",
        "docstring_parser.google",
    ],
    "apimd": ["apimd.parser", "apimd.loader"],
    "isort": ["isort.format", "isort.exceptions"],
    "sty": ["sty.lib", "sty.primitive"],
}

REPO_URL_DICT = {
    "apimd": "https://github.com/KmolYuan/apimd.git",
    "codetiming": "https://github.com/realpython/codetiming.git",
    "dataclasses_json": "https://github.com/lidatong/dataclasses-json.git",
    "docstring_parser": "https://github.com/rr-/docstring_parser.git",
    "flutes": "https://github.com/huzecong/flutes.git",
    "flutils": "https://gitlab.com/finite-loop/flutils.git",
    "httpie": "https://github.com/httpie/httpie.git",
    "isort": "https://github.com/PyCQA/isort.git",
    "mimesis": "https://github.com/lk-geimfari/mimesis.git",
    "py_backwards": "https://github.com/nvbn/py-backwards.git",
    "pymonet": "https://github.com/przemyslawjanpietrzak/pyMonet.git",
    "pypara": "https://github.com/vst/pypara.git",
    "semantic_release": "https://github.com/relekang/python-semantic-release.git",
    "string_utils": "https://github.com/daveoncode/python-string-utils.git",
    "pytutils": "https://github.com/akatrevorjay/pytutils.git",
    "sanic": "https://github.com/sanic-org/sanic.git",
    "sty": "https://github.com/feluxe/sty.git",
    "thonny": "https://github.com/thonny/thonny.git",
    "typesystem": "https://github.com/encode/typesystem.git",
    "pysnooper": "https://github.com/cool-RR/PySnooper.git",
    "ansible": "https://github.com/ansible/ansible.git",
    "cookiecutter": "https://github.com/cookiecutter/cookiecutter.git",
    "fastapi": "https://github.com/tiangolo/fastapi.git",
    "keras": "https://github.com/keras-team/keras.git",
    "luigi": "https://github.com/spotify/luigi.git",
    "pandas": "https://github.com/pandas-dev/pandas.git",
    "scrapy": "https://github.com/scrapy/scrapy.git",
    "spacy": "https://github.com/explosion/spaCy.git",
    "thef": "https://github.com/nvbn/thefuck.git",
    "tornado": "https://github.com/tornadoweb/tornado.git",
    "tqdm": "https://github.com/tqdm/tqdm.git",
    "youtube_dl": "https://github.com/ytdl-org/youtube-dl.git",
}

REPO_COMMIT_ID_DICT = {
    "apimd": "f32841b",
    "codetiming": "a7ad85a",
    "dataclasses_json": "3dc59e01",
    "docstring_parser": "a5dc2cd77",
    "flutes": "49647e4b",
    "flutils": "df0f84e1",
    "httpie": "bb36897",
    "isort": "a6222a8",
    "mimesis": "310092ce",
    "py_backwards": "8be3c4430",
    "pymonet": "f132cfa",
    "pypara": "7d705a54",
    "semantic_release": "3689157c2",
    "string_utils": "d903db3c2",
    "pytutils": "9813bb3",
    "sanic": "93a0246",
    "sty": "f99e9186",
    "thonny": "fb389f4",
    "typesystem": "6a9590c125",
    "pysnooper": "31bfc63",
    "ansible": "f00f123",
    "cookiecutter": "1c0b5b11",
    "fastapi": "864643e",
    "keras": "2c48a3b3",
    "luigi": "f2f631b",
    "pandas": "945c9ed",
    "scrapy": "61130c8",
    "spacy": "800737b",
    "thef": "0949d2e",
    "tornado": "2047e7a",
    "tqdm": "18d7aa4",
    "youtube_dl": "b224cf3",
}


def extract_instances(
    repo_name: str,
    repo_url: str,
    commit_id: str,
    module_list: List[str],
    baseline_temp_dir: str,
) -> List[dict]:
    """
    Extract instances from a repository by cloning it and inspecting modules.

    Args:
        repo_name: Name of the repository
        repo_url: URL of the repository
        commit_id: Git commit ID to checkout
        module_list: List of module names to inspect
        baseline_temp_dir: Temporary directory to clone the repo

    Returns:
        List of dictionaries containing module information
    """
    repo_new_path = os.path.join(baseline_temp_dir, repo_name)
    if os.path.exists(repo_new_path):
        repo_obj = Repo(repo_new_path)
    else:
        repo_obj = git.Repo.clone_from(repo_url, repo_new_path, no_checkout=True)
    repo_obj.git.checkout(commit_id)

    instances = []
    # Inspect modules
    for module in module_list:
        module_info = inspect_module(repo_new_path, module, repo_name)
        if module_info:
            instances.append(module_info)

    return instances


def inspect_module(repo_path: str, module_name: str, repo_name: str) -> dict:
    """
    Import a module and inspect it to extract source code and importable items.

    Args:
        repo_path: Path to the repository
        module_name: Full module name (e.g., 'package.module')
        repo_name: Name of the repository

    Returns:
        Dictionary containing module information including classes and functions
    """

    # Convert module name to file path
    module_path_parts = module_name.split(".")
    module_file_name = os.path.join(*module_path_parts) + ".py"

    # Try multiple common directory structures
    possible_paths = [
        os.path.join(repo_path, module_file_name),  # Direct: repo/package/module.py
        os.path.join(
            repo_path, "src", module_file_name
        ),  # Src layout: repo/src/package/module.py
        os.path.join(
            repo_path, "lib", module_file_name
        ),  # Lib layout: repo/lib/package/module.py
        os.path.join(
            repo_path, repo_name, module_file_name
        ),  # Package dir: repo/repo_name/package/module.py
    ]

    # Find the first existing path
    module_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            module_file_path = path
            break

    if module_file_path is None:
        return None

    module_info = {
        "repo_name": repo_name,
        "module_name": module_name,
        "module_path": module_file_path,
        "classes": [],
        "functions": [],
    }

    try:
        # Read source code
        with open(module_file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        module_info["source_code"] = source_code

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(module_name, module_file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Add potential source directories to sys.path
            paths_to_add = [
                repo_path,
                os.path.join(repo_path, "src"),
                os.path.join(repo_path, "lib"),
                os.path.dirname(module_file_path),
            ]
            for path in paths_to_add:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)

            try:
                spec.loader.exec_module(module)

                # Extract classes
                for name, obj in insp.getmembers(module, insp.isclass):
                    if obj.__module__ == module_name:
                        methods = []
                        for method_name, method_obj in insp.getmembers(
                            obj, insp.isfunction
                        ):
                            if not method_name.startswith("_"):
                                methods.append(method_name)
                        module_info["classes"].append(
                            {
                                "name": name,
                                "methods": methods,
                            }
                        )

                # Extract functions
                for name, obj in insp.getmembers(module, insp.isfunction):
                    if obj.__module__ == module_name and not name.startswith("_"):
                        module_info["functions"].append(name)
            finally:
                # Clean up sys.path
                for path in paths_to_add:
                    if path in sys.path:
                        sys.path.remove(path)
    except Exception as e:
        module_info["error"] = str(e)

    return module_info


class Codamosa(Data):

    def __init__(
        self,
        logger: Console,
        path: str,
        graph: Graph,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        llm_tokenizer: PreTrainedTokenizer,
        model_name: str,
        llm_model_name: str,
        debug: bool = False,
        baseline_prompt: str = "code",
        graph_sampling: bool = False,
        max_tokens: int = 512,
        gnn_mode: str = "node",
        raw_overwrite: bool = False,
        repo: str = None,
        **kwargs,
    ) -> None:
        self.name = "CodaMosa"
        super().__init__(
            name=self.name,
            path=path,
            logger=logger,
            graph=graph,
            feat_model=model,
            feat_tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            llm_model_name=llm_model_name,
            num_cpu=-1,
            debug=debug,
            model_name=model_name,
            baseline_prompt=baseline_prompt,
            graph_sampling=graph_sampling,
            max_tokens=max_tokens,
            gnn_mode=gnn_mode,
            repo=repo,
            **kwargs,
        )
        self.data_path = os.path.join(path, self.name)
        self.debug = debug
        self.data = None
        self.repo = repo
        self.raw_overwrite = raw_overwrite
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            self.logger.log(
                f"Data path not found, created a new one at {self.data_path}"
            )
        else:
            self.logger.log(f"Data path found at {self.data_path}")
            if os.path.exists(os.path.join(self.data_path, "data_processed.json")):
                with open(
                    os.path.join(self.data_path, "data_processed.json"), "r"
                ) as file:
                    self.data = json.load(file)
            else:
                if not os.path.exists(os.path.join(self.data_path, "data.jsonl")):
                    logger.log("data.jsonl not found, please crawl the data")
                else:
                    logger.log(
                        "Found data.jsonl file, but not processed. PLEASE RUN `process_raw`"
                    )
        if self.feat_model is not None:
            self.logger.log(
                f"Initialized {self.name} dataset, with model on device: {self.feat_model.device}"
            )

    def crawl(self) -> None:

        # Create a folder 'projects' under data_path
        projects_path = os.path.join(self.data_path, "projects")
        if not os.path.exists(projects_path):
            os.makedirs(projects_path)

        self.logger.log(f"Created projects directory at {projects_path}")

        raw_data_path = os.path.join(self.data_path, "raw_data.jsonl")
        # Check for prcessed repo
        if os.path.exists(raw_data_path):
            with open(raw_data_path, "r") as f:
                existing_repos = set()
                for line in f:
                    instance = json.loads(line)
                    existing_repos.add(instance["repo_name"])
        else:
            existing_repos = set()

        # For each repo in REPO_MODULE_DICT, clone the repo and get the instances from the modules
        for repo_name, module_list in REPO_MODULE_DICT.items():
            if repo_name in existing_repos:
                self.logger.log(
                    f"[green]Repository {repo_name} already processed, skipping[/green]"
                )
                continue
            repo_url = REPO_URL_DICT[repo_name]
            commit_id = REPO_COMMIT_ID_DICT[repo_name]
            self.logger.log(
                f"Cloning repository {repo_name} from {repo_url} at commit {commit_id}"
            )
            instances = extract_instances(
                repo_name=repo_name,
                repo_url=repo_url,
                commit_id=commit_id,
                module_list=module_list,
                baseline_temp_dir=projects_path,
            )
            self.logger.log(f"Extracted {len(instances)} modules from {repo_name}")
            with open(raw_data_path, "a") as f:
                for instance in instances:
                    f.write(json.dumps(instance) + "\n")

    def process_raw(self) -> None:
        self.logger.log("Processing raw data...")
        return

    def create_package_txt(self, data: dict) -> None:
        # Codamosa project comes with package.txt already
        pass

    def clean_up(self) -> None:
        # Go over each project and if package.txt is not created, remove that project
        pass

    def create_module_info(self) -> List[dict]:
        """
        Create a module info from the extracted data
        Each module info includes:
            - module_name_test_gen (e.g., path.to.module, without .py)
            - module_path
            - module_name
            - project
            - project_path
            - code_path to raw project
            - output_test_path
            - package_path
            - module_name_after_test_gen
            - graph_path
            - graph_name
            - module_name_coverage
        """
        module_infos = []
        for dat in self.data:
            project = dat["project"]
            package_path = dat["project_path"]
            for i, module in enumerate(dat["modules"]):
                module_info = {}
                module_info["module_name_full"] = f"{project}|{dat['module_name'][i]}"
                module_info["module_name"] = dat["module_name"][i]
                module_info["module_name_test_gen"] = module
                module_info["module_path"] = dat["module_path"][i]
                module_info["project"] = project
                module_info["project_path"] = package_path
                module_info["package_path"] = package_path
                module_info["code_path"] = os.path.join(package_path, project)
                module_info["output_test_path"] = os.path.join(package_path, "test")
                module_info["module_name_after_test_gen"] = (
                    f"test_{dat['module_name'][i]}.py"
                )
                module_info["graph_path"] = os.path.join(package_path, "graph")
                module_info["graph_name"] = f"{dat['module_name'][i]}.json"
                module_info["module_name_coverage"] = module.replace(".", "/")
                module_infos.append(module_info)
        return module_infos
