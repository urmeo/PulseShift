"""Build the merged hourly analysis panel."""

from pulseshift import panel


def main():
    df = panel.build_panel(write=True)
    active = df["active_hour"].sum()
    print(
        f"panel rows: {len(df)}  active hours: {active}  suppressed: {int(df['suppressed'].sum())}"
    )
    print(f"span: {df['ts_local'].min()} -> {df['ts_local'].max()}")


if __name__ == "__main__":
    main()
