# Template: Where the busiest channels are
# Group: Anatomy
# Mirrors: the Contacts and Map pages
# About: Event rates placed on the contacts' coordinates when the recording has an electrode file, on nilearn's glass brain when it is installed; otherwise each shaft as a row, contacts in order.

from onset_hfo.case.electrodes import channel_positions

rates = findings.set_index("channel")["rate_per_min"]

# %% With measured coordinates
if electrodes is not None and len(electrodes):
    frame = electrodes.rename(columns=str.lower)
    where = channel_positions(frame, list(rates.index))
    where = where[where["placed"]].assign(rate=lambda d: rates.reindex(d["channel"]).to_numpy())
    print(where.sort_values("rate", ascending=False).head(10).to_string(index=False))
    try:
        from nilearn import plotting

        display = plotting.plot_markers(where["rate"].to_numpy(),
                                        where[["x", "y", "z"]].to_numpy(float),
                                        node_cmap="viridis", display_mode="lzr",
                                        title="events / min")
        plt.show()
    except ImportError:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(where["x"], where["y"], where["z"], c=where["rate"], s=20 + 3 * where["rate"])
        plt.show()

# %% Without coordinates: each shaft as a row, contacts in order
else:
    print("This recording has no electrode positions: shafts drawn schematically.")
    shafts = pd.Series(rates.index).str.extract(r"^([A-Za-z]+)")[0]
    order = sorted(set(shafts))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1))
    for row, shaft in enumerate(order):
        names = [c for c, s in zip(rates.index, shafts, strict=True) if s == shaft]
        names.sort(key=lambda c: int("".join(ch for ch in c.split("-")[0] if ch.isdigit()) or 0))
        ax.scatter(range(len(names)), [row] * len(names), s=20 + 4 * rates[names],
                   c=rates[names], cmap="viridis", vmin=0, vmax=max(rates.max(), 1))
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("contact, in order along the shaft")
    ax.set_title("size and colour: events / min")
    fig.tight_layout()
    plt.show()
