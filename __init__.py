import os
import base64

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from embeddings import (
        run_embeddings_model,
        AIMv2_ARCHS,
    )


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


class AIMv2Embeddings(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="apple_aim_embeddings",  # required

            # The display name of the operator
            label="AIMv2 Embeddings",  # required

            # A description for the operator
            description="Compute embeddings using AIMv2 Models",

            icon="/assets/apple-logo.svg",

            # Whether the operator supports immediate and/or delegated execution
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            )


    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        model_dropdown = types.Dropdown(label="Select the embedding model you want to use")

        for arch in AIMv2_ARCHS:
            model_dropdown.add_choice(arch, label=arch)

        inputs.enum(
            "model_name",
            choices=AIMv2_ARCHS,
            label="Embedding Model",
            description="Choose the AIMv2 embedding model you want to use:",
            view=model_dropdown,
            required=True
        )

        embedding_types = types.RadioGroup()

        embedding_types.add_choice(
            "cls", 
            label="Class token embedding",
            description="A single embedding vector derived from special classification token. Represents the global semantic context of an image."
            )
        
        embedding_types.add_choice(
            "mean", 
            label="Mean pooling embedding",
            description="An embedding vector computed by averaging the representations of all image patches. Captures distributed contextual information across the entire input."
            )
        
        inputs.enum(
            "embedding_types",
            embedding_types.values(),
            view=embedding_types,
            caption="Which embedding approach do you want to use?",
            required=True
        )

        inputs.str(
            "emb_field",
            label="Embeddings Field",
            description="Name field to store the embeddings in",
            required=True,
            )
        
        _execution_mode(ctx, inputs)

        inputs.view_target(ctx)


        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        model_name = ctx.params.get("model_name")
        embedding_types = ctx.params.get("embedding_types")
        emb_field = ctx.params.get("emb_field")

        run_embeddings_model(
            dataset=view,
            model_name=model_name,
            emb_field=emb_field,
            embedding_types=embedding_types
            )
        
        ctx.ops.reload_dataset()


def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(AIMv2Embeddings)