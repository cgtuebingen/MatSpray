from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.neilf_deferred import render_neilf_deferred
from gaussian_renderer.neilf_deferred_importance import render_neilf_deferred as render_neilf_deferred_importance


render_fn_dict = {
    "render": render,
    "neilf": render_neilf,
    "neilf_deferred": render_neilf_deferred,
    "neilf_deferred_importance": render_neilf_deferred_importance,
}