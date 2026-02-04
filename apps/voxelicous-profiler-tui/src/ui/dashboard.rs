//! Main profiler dashboard UI.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame,
};

use voxelicous_profiler::ProfilerSnapshot;

use crate::client::ConnectionState;

/// Dashboard state and rendering.
pub struct Dashboard {
    host: String,
    port: u16,
}

impl Dashboard {
    /// Create a new dashboard.
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
        }
    }

    /// Render the dashboard.
    pub fn render(
        &self,
        frame: &mut Frame,
        connection_state: ConnectionState,
        snapshot: Option<&ProfilerSnapshot>,
    ) {
        let area = frame.area();

        // Main layout: header, content, footer
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Header
                Constraint::Length(3), // Frame info
                Constraint::Min(10),   // Stats table
                Constraint::Length(3), // Queue info
                Constraint::Length(3), // Footer
            ])
            .split(area);

        self.render_header(frame, sections[0], connection_state, snapshot);
        self.render_frame_info(frame, sections[1], snapshot);
        self.render_stats_table(frame, sections[2], snapshot);
        self.render_queue_info(frame, sections[3], snapshot);
        self.render_footer(frame, sections[4]);
    }

    fn render_header(
        &self,
        frame: &mut Frame,
        area: Rect,
        state: ConnectionState,
        snapshot: Option<&ProfilerSnapshot>,
    ) {
        let (status_text, status_color) = match state {
            ConnectionState::Connected => ("Connected", Color::Green),
            ConnectionState::Connecting => ("Connecting...", Color::Yellow),
            ConnectionState::Disconnected => ("Disconnected", Color::Red),
        };

        let frame_num = snapshot.map_or(0, |s| s.frame_number);

        let title = Line::from(vec![
            Span::styled(
                " Voxelicous Profiler ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("| "),
            Span::styled(
                format!("{}:{}", self.host, self.port),
                Style::default().fg(Color::White),
            ),
            Span::raw(" | "),
            Span::styled(status_text, Style::default().fg(status_color)),
            Span::raw(" | Frame: "),
            Span::styled(format!("{}", frame_num), Style::default().fg(Color::Yellow)),
        ]);

        let header = Paragraph::new(title).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan)),
        );

        frame.render_widget(header, area);
    }

    fn render_frame_info(
        &self,
        frame: &mut Frame,
        area: Rect,
        snapshot: Option<&ProfilerSnapshot>,
    ) {
        let (fps, frame_time, update_time, gpu_sync_time, render_time, submit_time, present_time) =
            snapshot.map_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |s| {
                let update = s
                    .categories
                    .iter()
                    .find(|c| c.category == voxelicous_profiler::EventCategory::FrameUpdate)
                    .map_or(0.0, |c| c.avg_ms());
                let gpu_sync = s
                    .categories
                    .iter()
                    .find(|c| c.category == voxelicous_profiler::EventCategory::GpuSync)
                    .map_or(0.0, |c| c.avg_ms());
                let render = s
                    .categories
                    .iter()
                    .find(|c| c.category == voxelicous_profiler::EventCategory::FrameRender)
                    .map_or(0.0, |c| c.avg_ms());
                let submit = s
                    .categories
                    .iter()
                    .find(|c| c.category == voxelicous_profiler::EventCategory::GpuSubmit)
                    .map_or(0.0, |c| c.avg_ms());
                let present = s
                    .categories
                    .iter()
                    .find(|c| c.category == voxelicous_profiler::EventCategory::FramePresent)
                    .map_or(0.0, |c| c.avg_ms());
                (
                    s.fps,
                    s.frame_time_ms,
                    update,
                    gpu_sync,
                    render,
                    submit,
                    present,
                )
            });

        let fps_color = if fps >= 60.0 {
            Color::Green
        } else if fps >= 30.0 {
            Color::Yellow
        } else {
            Color::Red
        };

        let info = Line::from(vec![
            Span::raw(" FPS: "),
            Span::styled(
                format!("{:.1}", fps),
                Style::default().fg(fps_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | Frame: "),
            Span::styled(
                format!("{:.1}ms", frame_time),
                Style::default().fg(Color::White),
            ),
            Span::raw(" (Upd: "),
            Span::styled(
                format!("{:.1}", update_time),
                Style::default().fg(Color::Blue),
            ),
            Span::raw(" Sync: "),
            Span::styled(
                format!("{:.1}", gpu_sync_time),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(" Rnd: "),
            Span::styled(
                format!("{:.1}", render_time),
                Style::default().fg(Color::Magenta),
            ),
            Span::raw(" Sub: "),
            Span::styled(
                format!("{:.1}", submit_time),
                Style::default().fg(Color::Green),
            ),
            Span::raw(" Prs: "),
            Span::styled(
                format!("{:.1}", present_time),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(")"),
        ]);

        let widget = Paragraph::new(info).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Frame Timing ")
                .border_style(Style::default().fg(Color::Blue)),
        );

        frame.render_widget(widget, area);
    }

    fn render_stats_table(
        &self,
        frame: &mut Frame,
        area: Rect,
        snapshot: Option<&ProfilerSnapshot>,
    ) {
        let header_cells = ["Category", "Count", "Avg", "Min", "Max", "Total"]
            .iter()
            .map(|h| Cell::from(*h).style(Style::default().fg(Color::Yellow)));
        let header = Row::new(header_cells)
            .style(Style::default().add_modifier(Modifier::BOLD))
            .height(1);

        let rows: Vec<Row> = snapshot.map_or_else(Vec::new, |s| {
            s.categories
                .iter()
                .filter(|c| {
                    // Skip the overall Frame category (redundant with header)
                    !matches!(c.category, voxelicous_profiler::EventCategory::Frame)
                })
                .map(|stat| {
                    let color = match stat.category {
                        voxelicous_profiler::EventCategory::FrameUpdate => Color::Blue,
                        voxelicous_profiler::EventCategory::GpuSync => Color::Yellow,
                        voxelicous_profiler::EventCategory::FrameRender => Color::Magenta,
                        voxelicous_profiler::EventCategory::GpuSubmit => Color::Green,
                        voxelicous_profiler::EventCategory::FramePresent => Color::Cyan,
                        voxelicous_profiler::EventCategory::ClipmapPageBuild => Color::LightGreen,
                        voxelicous_profiler::EventCategory::ClipmapEncode => Color::LightCyan,
                        voxelicous_profiler::EventCategory::GpuClipmapUpload => Color::LightMagenta,
                        voxelicous_profiler::EventCategory::GpuClipmapUnload => Color::LightRed,
                        voxelicous_profiler::EventCategory::ClipmapUpdate => Color::LightBlue,
                        _ => Color::White,
                    };

                    Row::new(vec![
                        Cell::from(stat.category.name()).style(Style::default().fg(color)),
                        Cell::from(format!("{}", stat.count)),
                        Cell::from(format!("{:.2}ms", stat.avg_ms())),
                        Cell::from(format!("{:.2}ms", stat.min_ms())),
                        Cell::from(format!("{:.2}ms", stat.max_ms())),
                        Cell::from(format!("{:.2}ms", stat.total_ms())),
                    ])
                })
                .collect()
        });

        let widths = [
            Constraint::Length(15),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(12),
        ];

        let table = Table::new(rows, widths)
            .header(header)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Timing Statistics ")
                    .border_style(Style::default().fg(Color::Green)),
            )
            .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        frame.render_widget(table, area);
    }

    fn render_queue_info(
        &self,
        frame: &mut Frame,
        area: Rect,
        snapshot: Option<&ProfilerSnapshot>,
    ) {
        let queues = snapshot.map_or_else(Default::default, |s| s.queues);

        let info = Line::from(vec![
            Span::raw(" Page Uploads: "),
            Span::styled(
                format!("{}", queues.pending_page_uploads),
                Style::default().fg(Color::Magenta),
            ),
            Span::raw(" | Page Unloads: "),
            Span::styled(
                format!("{}", queues.pending_page_unloads),
                Style::default().fg(Color::Red),
            ),
            Span::raw(" | Build Queue: "),
            Span::styled(
                format!("{}", queues.pending_page_builds),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(" | Resident Pages: "),
            Span::styled(
                format!("{}", queues.resident_pages),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(" | GPU Pages: "),
            Span::styled(
                format!("{}", queues.gpu_pages),
                Style::default().fg(Color::Green),
            ),
        ]);

        let widget = Paragraph::new(info).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Queues ")
                .border_style(Style::default().fg(Color::Yellow)),
        );

        frame.render_widget(widget, area);
    }

    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer = Paragraph::new(Line::from(vec![
            Span::styled(" [Q] ", Style::default().fg(Color::Yellow)),
            Span::raw("Quit"),
            Span::raw("  "),
            Span::styled("[R] ", Style::default().fg(Color::Yellow)),
            Span::raw("Reset Stats"),
            Span::raw("  "),
            Span::styled("[C] ", Style::default().fg(Color::Yellow)),
            Span::raw("Reconnect"),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );

        frame.render_widget(footer, area);
    }
}
