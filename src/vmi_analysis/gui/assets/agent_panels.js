import { Type } from 'typebox';

export default function (pi) {
    pi.registerTool({
        name: 'vmi_panel',
        label: 'VMI panel',
        description: 'Inspect workbench operations or prepare a real panel job for the user to approve. '
            + 'Use describe first to learn the named parameters and defaults. '
            + 'Use inspect with the view ID supplied by a read-only panel preview. '
            + 'Use status with the job ID returned after approval. Use wait to follow a running job. '
            + 'If wait returns a checkpoint, tell the user what to confirm in the app. '
            + 'Pass completed job IDs to subsequent stages. On a shared server, paths inside the container '
            + 'are /workspace and /datasets. The browser displays resolved paths and parameters before submission.',
        parameters: Type.Object({
            action: Type.Union(['describe', 'inspect', 'propose', 'status', 'wait'].map(action => Type.Literal(action))),
            operation: Type.Optional(Type.String()),
            parameters: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
            job_id: Type.Optional(Type.String()),
            view_id: Type.Optional(Type.String({description: 'Use the supplied view ID when proposing an operation from the active panel view.'})),
        }),
        async execute(_id, params, signal, _onUpdate, ctx) {
            const payload = JSON.stringify(params);
            if (payload.length > 8192) throw new Error('Keep the panel request under 8192 characters.');
            const value = await ctx.ui.input('VMI panel operation', payload, {
                timeout: 300000, signal,
            });
            return {
                content: [{ type: 'text', text: value ?? 'The request was dismissed or timed out.' }],
                details: {},
            };
        },
    });
}
