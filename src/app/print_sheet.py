"""Printable consultation handout component (A4 Polish sheet with U.D.A.R. box and print CSS)."""

from datetime import date
from typing import Dict, Any, Optional
import streamlit as st


def generate_printable_html(
    patient: Dict[str, Any],
    prob: float,
    optimal_th: float,
    tier: Dict[str, str],
    vasc_age: float,
    delta_age: float,
    rejuv_potential: float,
    sim_patient: Dict[str, Any],
    sim_prob: float,
    arr: float,
    rrr: float,
    consultation_date: Optional[str] = None,
) -> str:
    """Generate self-contained HTML for A4 printable dual-zone consultation sheet."""
    if consultation_date is None:
        consultation_date = date.today().strftime("%d.%m.%Y")

    # Polish translations for patient demographic fields
    gender_map = {
        "Male": "Mężczyzna",
        "Female": "Kobieta",
        "Other": "Inna",
    }
    gender_pl = gender_map.get(str(patient.get("gender", "")), str(patient.get("gender", "")))

    smoking_map = {
        "smokes": "Pali regularnie",
        "formerly smoked": "Palił w przeszłości",
        "never smoked": "Nigdy nie palił",
        "Unknown": "Brak danych / nieznany",
    }
    smoking_current_pl = smoking_map.get(
        str(patient.get("smoking_status", "")), str(patient.get("smoking_status", ""))
    )
    smoking_target_pl = smoking_map.get(
        str(sim_patient.get("smoking_status", "")), str(sim_patient.get("smoking_status", ""))
    )
    if sim_patient.get("smoking_status") == "never smoked" and patient.get("smoking_status") != "never smoked":
        smoking_target_pl = "Zaprzestanie palenia (abstynencja)"

    # Hypertension status
    htn_current = patient.get("hypertension", 0) == 1
    htn_target = sim_patient.get("hypertension", 0) == 1
    htn_current_pl = "Nadciśnienie tętnicze (niekontrolowane)" if htn_current else "Prawidłowe ciśnienie tętnicze"
    htn_target_pl = "Skuteczna kontrola RR (<130/80 mmHg)" if not htn_target else "Niekontrolowane"

    # Vascular age framing
    if delta_age > 0:
        vasc_status_text = f"+{delta_age:.1f} lat powyżej wieku metrykalnego"
        rejuv_text = f"-{rejuv_potential:.1f} lat (potencjał rewitalizacji naczyń)"
    else:
        vasc_status_text = "Optymalny stan naczyń"
        rejuv_text = "Profil optymalny (utrzymanie protekcji)"

    # Screening cutoff status
    is_screen_positive = prob >= optimal_th
    cutoff_pct = optimal_th * 100.0
    cutoff_badge = (
        f'<span style="background: #fee2e2; color: #dc2626; padding: 2px 8px; border-radius: 4px; font-weight: 700; border: 1px solid #fca5a5;">POWYŻEJ PROGU SKRININGOWEGO (Wynik dodatni, p*={cutoff_pct:.1f}%)</span>'
        if is_screen_positive
        else f'<span style="background: #dcfce7; color: #16a34a; padding: 2px 8px; border-radius: 4px; font-weight: 700; border: 1px solid #86efac;">PONIŻEJ PROGU SKRININGOWEGO (Wynik ujemny, p*={cutoff_pct:.1f}%)</span>'
    )

    tier_name = tier.get("tier_pl", tier.get("tier", "Umiarkowane"))
    tier_color = tier.get("color", "#ca8a04")
    tier_bg = tier.get("bg", "#fef9c3")
    tier_border = tier.get("border", "#fde047")
    recommendation = tier.get("recommendation", "")

    html = f"""
<div id="printable-consultation">
    <style>
    /* Injected print stylesheet isolating #printable-consultation */
    @media print {{
        body * {{
            visibility: hidden;
        }}
        #printable-consultation, #printable-consultation * {{
            visibility: visible;
        }}
        #printable-consultation {{
            position: absolute;
            left: 0;
            top: 0;
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            box-shadow: none !important;
            box-sizing: border-box !important;
            background: #ffffff !important;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }}
        .no-print, header, footer, [data-testid="stSidebar"], [data-testid="stHeader"] {{
            display: none !important;
        }}
        @page {{
            size: A4 portrait;
            margin: 10mm;
        }}
    }}

    /* Screen preview styling */
    #printable-consultation {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
        max-width: 900px;
        margin: 10px auto;
        box-sizing: border-box;
        page-break-after: avoid;
        break-after: avoid;
    }}

    #printable-consultation h1, #printable-consultation h2, #printable-consultation h3 {{
        margin: 0;
        padding: 0;
    }}

    .sheet-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        border-bottom: 2px solid #0284c7;
        padding-bottom: 10px;
        margin-bottom: 12px;
    }}
    .sheet-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.2px;
    }}
    .sheet-subtitle {{
        font-size: 0.8rem;
        color: #0369a1;
        font-weight: 600;
        margin-top: 2px;
    }}
    .sheet-meta {{
        text-align: right;
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.35;
    }}

    /* Zone container structure: 70% Patient Education / 30% Physician EHR */
    .zone-patient {{
        margin-bottom: 14px;
    }}
    .zone-physician {{
        border-top: 2px dashed #94a3b8;
        padding-top: 12px;
        margin-top: 12px;
        background: #f8fafc;
        border-radius: 8px;
        padding: 12px 14px;
    }}
    .zone-badge {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 8px;
    }}
    .badge-patient {{
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }}
    .badge-physician {{
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }}

    /* Demographics & Baseline grid */
    .demo-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
        margin-bottom: 12px;
    }}
    .demo-cell {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }}
    .demo-label {{
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
    }}
    .demo-val {{
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 2px;
    }}
    .demo-sub {{
        font-size: 0.7rem;
        color: #475569;
        margin-top: 1px;
    }}

    /* Targets table */
    .sheet-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.78rem;
        margin-bottom: 12px;
    }}
    .sheet-table th {{
        background: #f1f5f9;
        color: #334155;
        font-weight: 700;
        text-align: left;
        padding: 6px 10px;
        border: 1px solid #e2e8f0;
    }}
    .sheet-table td {{
        padding: 5px 10px;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }}
    .sheet-table tr:nth-child(even) {{
        background: #fafafa;
    }}

    /* Risk Reduction Highlight */
    .reduction-box {{
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 8px;
        padding: 8px 12px;
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-bottom: 12px;
        text-align: center;
    }}
    .reduction-item {{
        flex: 1;
    }}
    .reduction-val {{
        font-size: 1.25rem;
        font-weight: 800;
        color: #047857;
    }}
    .reduction-label {{
        font-size: 0.72rem;
        font-weight: 600;
        color: #065f46;
    }}

    /* Prominent U.D.A.R. warning box */
    .udar-container {{
        background: #fff5f5;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }}
    .udar-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }}
    .udar-title {{
        font-size: 0.92rem;
        font-weight: 800;
        color: #b91c1c;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .udar-sos {{
        font-size: 0.78rem;
        font-weight: 800;
        background: #dc2626;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 999px;
    }}
    .udar-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }}
    .udar-card {{
        background: #ffffff;
        border: 1px solid #fecaca;
        border-radius: 6px;
        padding: 6px 8px;
    }}
    .udar-letter {{
        font-size: 1.1rem;
        font-weight: 900;
        color: #dc2626;
        display: inline-block;
        margin-right: 4px;
    }}
    .udar-name {{
        font-size: 0.78rem;
        font-weight: 800;
        color: #7f1d1d;
    }}
    .udar-desc {{
        font-size: 0.68rem;
        color: #4b5563;
        line-height: 1.25;
        margin-top: 2px;
    }}

    /* Physician section styles */
    .physician-grid {{
        display: grid;
        grid-template-columns: 1.4fr 1fr;
        gap: 12px;
        font-size: 0.75rem;
    }}
    .physician-plan {{
        line-height: 1.4;
        color: #334155;
    }}
    .signature-block {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px 10px;
        min-height: 85px;
    }}
    .sig-line {{
        border-bottom: 1px dotted #94a3b8;
        margin-top: 22px;
    }}
    .sig-caption {{
        font-size: 0.68rem;
        color: #64748b;
        text-align: center;
        margin-top: 4px;
    }}
    </style>

    <!-- HEADER STRIP -->
    <div class="sheet-header">
        <div>
            <div class="sheet-title">🏥 KARTA KONSULTACYJNA PREWENCJI UDARU MÓZGU</div>
            <div class="sheet-subtitle">System Wspomagania Decyzji Klinicznych (CDSS) • Profilaktyka Pierwotna i Wtórna</div>
        </div>
        <div class="sheet-meta">
            <div><strong>Data konsultacji:</strong> {consultation_date}</div>
            <div><strong>Standard:</strong> PTK / AHA / Spec #21</div>
            <div><strong>ID Rekordu:</strong> PL-POZ-STROKE-{patient.get('age', 0):.0f}</div>
        </div>
    </div>

    <!-- ZONE 1: PATIENT EDUCATION (70%) -->
    <div class="zone-patient">
        <div class="zone-badge badge-patient">Część I: Edukacja Pacjenta i Indywidualny Plan Ochrony Naczyń (70%)</div>

        <!-- Demographics & Metrics -->
        <div class="demo-grid">
            <div class="demo-cell">
                <div class="demo-label">Pacjent / Metryka</div>
                <div class="demo-val">{patient.get('age', 0):.0f} lat</div>
                <div class="demo-sub">{gender_pl}</div>
            </div>
            <div class="demo-cell">
                <div class="demo-label">Ryzyko 5-letnie</div>
                <div class="demo-val" style="color: {tier_color};">{prob * 100:.1f}%</div>
                <div class="demo-sub">
                    <span style="background:{tier_bg}; color:{tier_color}; font-weight:700; padding:1px 6px; border-radius:4px; border:1px solid {tier_border}; font-size:0.68rem;">
                        {tier_name.upper()}
                    </span>
                </div>
            </div>
            <div class="demo-cell">
                <div class="demo-label">Wiek Naczyniowy</div>
                <div class="demo-val">{vasc_age:.1f} lat</div>
                <div class="demo-sub">{vasc_status_text}</div>
            </div>
            <div class="demo-cell">
                <div class="demo-label">Potencjał Rewitalizacji</div>
                <div class="demo-val" style="color: #0284c7;">{rejuv_text.split()[0]}</div>
                <div class="demo-sub">{rejuv_text}</div>
            </div>
        </div>

        <!-- Biomarker Targets Table -->
        <table class="sheet-table">
            <thead>
                <tr>
                    <th style="width: 28%;">Czynnik Ryzyka / Biomarker</th>
                    <th style="width: 22%;">Wartość Bieżąca</th>
                    <th style="width: 26%;">Cel Terapeutyczny</th>
                    <th style="width: 24%;">Status Interwencji</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Glikemia na czczo</strong></td>
                    <td>{patient.get('avg_glucose_level', 0):.1f} mg/dL</td>
                    <td><strong>{sim_patient.get('avg_glucose_level', 0):.0f} mg/dL</strong> (norma &lt; 100)</td>
                    <td>{'⚠️ Podwyższona' if patient.get('avg_glucose_level', 0) > 100 else '✅ W normie'}</td>
                </tr>
                <tr>
                    <td><strong>Wskaźnik Masy Ciała (BMI)</strong></td>
                    <td>{patient.get('bmi', 0):.1f} kg/m²</td>
                    <td><strong>{sim_patient.get('bmi', 0):.1f} kg/m²</strong> (norma 18.5 - 24.9)</td>
                    <td>{'⚠️ Nadwaga/Otyłość' if patient.get('bmi', 0) >= 25.0 else '✅ W normie'}</td>
                </tr>
                <tr>
                    <td><strong>Status Palenia Tytoniu</strong></td>
                    <td>{smoking_current_pl}</td>
                    <td><strong>{smoking_target_pl}</strong></td>
                    <td>{'🚨 Wskazane zaprzestanie' if patient.get('smoking_status') == 'smokes' else '✅ Zgodny z celem'}</td>
                </tr>
                <tr>
                    <td><strong>Kontrola Ciśnienia Krwi</strong></td>
                    <td>{htn_current_pl}</td>
                    <td><strong>{htn_target_pl}</strong></td>
                    <td>{'⚠️ Wymaga optymalizacji' if htn_current else '✅ Prawidłowe'}</td>
                </tr>
            </tbody>
        </table>

        <!-- Projected Risk Reduction -->
        <div class="reduction-box">
            <div class="reduction-item">
                <div class="reduction-val">{sim_prob * 100:.1f}%</div>
                <div class="reduction-label">Docelowe Ryzyko po Interwencji</div>
            </div>
            <div class="reduction-item" style="border-left: 1px solid #a7f3d0; border-right: 1px solid #a7f3d0;">
                <div class="reduction-val">+{arr:.1f}%</div>
                <div class="reduction-label">Bezwzględna Redukcja Ryzyka (ARR)</div>
            </div>
            <div class="reduction-item">
                <div class="reduction-val">{rrr:.0f}%</div>
                <div class="reduction-label">Względna Redukcja Ryzyka (RRR)</div>
            </div>
        </div>

        <!-- Prominently Box: Polish U.D.A.R. / F.A.S.T. Warning -->
        <div class="udar-container">
            <div class="udar-header">
                <div class="udar-title">
                    🚨 TEST U.D.A.R. (F.A.S.T.) — ROZPOZNAWANIE NAGŁYCH OBJAWÓW UDARU MÓZGU
                </div>
                <div class="udar-sos">TEL: 112 (POGOTOWIE)</div>
            </div>
            <div class="udar-grid">
                <div class="udar-card">
                    <div><span class="udar-letter">U</span><span class="udar-name">Usta wykrzywione</span></div>
                    <div class="udar-desc">Asymetria kącika ust, paraliż mięśni twarzy, opadanie wargi przy uśmiechu.</div>
                </div>
                <div class="udar-card">
                    <div><span class="udar-letter">D</span><span class="udar-name">Dłoń / ręka opada</span></div>
                    <div class="udar-desc">Nagły niedowład kończyny górnej, brak siły w uniesieniu rąk do góry.</div>
                </div>
                <div class="udar-card">
                    <div><span class="udar-letter">A</span><span class="udar-name">Artykulacja utrudniona</span></div>
                    <div class="udar-desc">Bełkotliwa, zamazana mowa, trudność z powtórzeniem prostego zdania.</div>
                </div>
                <div class="udar-card">
                    <div><span class="udar-letter">R</span><span class="udar-name">Rozstrzygaj i dzwoń!</span></div>
                    <div class="udar-desc">Natychmiast wybierz numer <strong>112</strong>. Liczy się każda minuta (okno terapeutyczne)!</div>
                </div>
            </div>
        </div>
    </div>

    <!-- ZONE 2: PHYSICIAN EHR SUMMARY & SIGNATURE BLOCK (30%) -->
    <div class="zone-physician">
        <div class="zone-badge badge-physician">Część II: Podsumowanie Kliniczne dla EHR & Autoryzacja Lekarska (30%)</div>
        <div class="physician-grid">
            <div class="physician-plan">
                <div style="margin-bottom: 4px;">
                    <strong>Ocena Stratyfikacji CDSS:</strong> {cutoff_badge}
                </div>
                <div style="margin-bottom: 4px;">
                    <strong>Zalecenie Diagnostyczno-Terapeutyczne:</strong> {recommendation}
                </div>
                <div>
                    <strong>Plan Postępowania:</strong>
                    Niefarmakologiczne: dieta śródziemnomorska/DASH, redukcja masy ciała, min. 150 min/tydz. wysiłku aerobowego, abstynencja nikotynowa.
                    Farmakologiczne: optymalizacja leczenia hipotensyjnego (&lt;130/80 mmHg), lipidogram i statynoterapia wg ryzyka sercowo-naczyniowego, kontrola glikemii (HbA1c &lt; 7.0%).
                </div>
            </div>
            <div class="signature-block">
                <div>
                    <div style="font-size: 0.72rem; color: #475569;">
                        <strong>Data badania / wystawienia:</strong> {consultation_date}
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b; margin-top: 2px;">
                        Status walidacji: Kwalifikacja POZ
                    </div>
                </div>
                <div>
                    <div class="sig-line"></div>
                    <div class="sig-caption">Pieczątka i podpis lekarza prowadzącego</div>
                </div>
            </div>
        </div>
    </div>
</div>
"""
    return html


def render_printable_consultation(
    patient: Dict[str, Any],
    prob: float,
    optimal_th: float,
    tier: Dict[str, str],
    vasc_age: float,
    delta_age: float,
    rejuv_potential: float,
    sim_patient: Dict[str, Any],
    sim_prob: float,
    arr: float,
    rrr: float,
    consultation_date: Optional[str] = None,
) -> None:
    """Render the print trigger button and the consultation sheet preview container."""
    st.markdown(
        """
        <hr style="margin: 24px 0 16px 0; border: none; border-top: 1px solid #bae6fd;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #0f172a;">
                    📄 Karta Konsultacyjna Prewencji Udaru Mózgu (Format A4 / PDF)
                </div>
                <div style="font-size: 0.84rem; color: #475569;">
                    Dwustrefowy dokument konsultacyjny (70% edukacja pacjenta z testem U.D.A.R., 30% podsumowanie EHR dla lekarza).
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1-Click print trigger button using st.components.v1.html to invoke window.parent.print()
    button_html = """
    <div style="margin: 0; padding: 4px 0;">
        <button onclick="try { window.parent.print(); } catch (e) { window.print(); }"
            style="
                background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px 22px;
                font-size: 14px;
                font-weight: 700;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35);
                display: inline-flex;
                align-items: center;
                gap: 8px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            "
            onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 16px rgba(2, 132, 199, 0.45)';"
            onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(2, 132, 199, 0.35)';">
            🖨️ Drukuj podsumowanie konsultacji (PDF/A4)
        </button>
        <span style="margin-left: 14px; font-size: 13px; color: #64748b; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            Izoluje arkusz <code>#printable-consultation</code> i ukrywa elementy interfejsu (CSS <code>@media print</code>).
        </span>
    </div>
    """
    st.components.v1.html(button_html, height=52)

    # Render the consultation HTML preview
    html_content = generate_printable_html(
        patient=patient,
        prob=prob,
        optimal_th=optimal_th,
        tier=tier,
        vasc_age=vasc_age,
        delta_age=delta_age,
        rejuv_potential=rejuv_potential,
        sim_patient=sim_patient,
        sim_prob=sim_prob,
        arr=arr,
        rrr=rrr,
        consultation_date=consultation_date,
    )
    st.markdown(html_content, unsafe_allow_html=True)
