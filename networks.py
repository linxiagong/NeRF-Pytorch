import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_views=3,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False,
                 net_chunk: int = None):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.net_chunk = net_chunk

    def forward(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def batchify_predict(self, input_pts, input_views):
        if self.net_chunk is None:
            return self.forward(input_pts, input_views)

        num_pts = input_pts.shape[0]
        chunk = int(self.net_chunk)
        return torch.cat([
            self.forward(input_pts=input_pts[i:i + chunk],
                         input_views=input_views[i:i + chunk] if input_views is not None else None)
            for i in range(0, num_pts, chunk)
        ], 0)


# # Positional encoding (section 5.1)
# class Embedder0:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()

#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x: x)
#             out_dim += d

#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']

#         if self.kwargs['log_sampling']:
#             freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
#                 out_dim += d

#         self.embed_fns = embed_fns
#         self.out_dim = out_dim

#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# def get_embedder(multires, i=0):
#     if i == -1:
#         return nn.Identity(), 3

#     embed_kwargs = {
#         'include_input': True,
#         'input_dims': 3,
#         'max_freq_log2': multires - 1,
#         'num_freqs': multires,
#         'log_sampling': True,
#         'periodic_fns': [torch.sin, torch.cos],
#     }

#     embedder_obj = Embedder0(**embed_kwargs)
#     embed = lambda x, eo=embedder_obj: eo.embed(x)
#     return embed, embedder_obj.out_dim


class Embedder(nn.Module):
    def __init__(self,
                 input_dims,
                 include_input,
                 multires,
                 log_sampling,
                 periodic_fns=[torch.sin, torch.cos],
                 **kwargs):
        super(Embedder, self).__init__()
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = multires - 1
        self.num_freqs = multires
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

    def forward(self, inputs):
        return self.embed(inputs)


class NeRFFull(nn.Module):
    """Full NeRF Model"""
    def __init__(self, model_params: dict) -> None:
        super().__init__()
        # positional embedding
        if model_params["pts_embedder"]["use_pos_embed"]:
            self.embedder_pts = Embedder(**model_params["pts_embedder"])
            pts_dim = self.embedder_pts.out_dim
        else:
            self.embedder_pts = nn.Identity()
            pts_dim = model_params["pts_embedder"]["input_dims"]

        self._use_viewdirs = model_params["use_viewdirs"]
        viewdirs_dim = 0
        if self._use_viewdirs:
            if model_params["viewdirs_embedder"]["use_pos_embed"]:
                self.embedder_viewdirs = Embedder(**model_params["viewdirs_embedder"])
                viewdirs_dim = self.embedder_viewdirs.out_dim
            else:
                self.embedder_viewdirs = nn.Identity()
                viewdirs_dim = model_params["viewdirs_embedder"]["input_dims"]

        # nerf nets
        net_chunk = model_params.get("net_chunk", None)
        self.nerf_nets = nn.ModuleDict({
            'coarse':
            NeRF(D=model_params["coarse_nerf"]["depth"],
                 W=model_params["coarse_nerf"]["width"],
                 input_ch=pts_dim,
                 input_ch_views=viewdirs_dim,
                 output_ch=4,
                 skips=model_params["coarse_nerf"]["skips"],
                 use_viewdirs=self._use_viewdirs,
                 net_chunk=net_chunk),
        })
        if model_params.get("use_fine_net", False):
            self.nerf_nets['fine'] = NeRF(D=model_params["fine_nerf"]["depth"],
                                          W=model_params["fine_nerf"]["width"],
                                          input_ch=pts_dim,
                                          input_ch_views=viewdirs_dim,
                                          output_ch=4,
                                          skips=model_params["fine_nerf"]["skips"],
                                          use_viewdirs=self._use_viewdirs,
                                          net_chunk=net_chunk)

    def forward(self, pts: torch.Tensor, viewdirs=None, model_name: str = 'coarse'):
        """
        pts: [B, N, 3]
        viewdirs: [B, 3]
        """
        pts_flat = torch.reshape(pts, (-1, pts.shape[-1]))
        pts_embedded = self.embedder_pts(pts_flat)

        viewdirs_embedded = None
        if self._use_viewdirs:
            if viewdirs is None:
                raise Exception('viewdirs is not given!')
            viewdirs_flat = viewdirs[:, None].expand(pts.shape)
            viewdirs_flat = torch.reshape(viewdirs_flat, (-1, viewdirs_flat.shape[-1]))
            viewdirs_embedded = self.embedder_viewdirs(viewdirs_flat)

        if model_name not in self.nerf_nets:
            model_name = 'coarse'
        # outputs_flat = self.nerf_nets[model_name](pts_embedded, viewdirs_embedded)
        outputs_flat = self.nerf_nets[model_name].batchify_predict(pts_embedded, viewdirs_embedded)
        outputs = torch.reshape(outputs_flat, (*pts.shape[:-1], outputs_flat.shape[-1]))
        return outputs
