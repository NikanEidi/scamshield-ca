"""
Generates data/scam_dataset.csv by combining message templates with
varying details (bank names, amounts, fake domains, phone numbers, etc.)
so we get a large, diverse dataset without hand-typing every row.

Templates are based on patterns documented by real Canadian sources
(CIBC fraud-examples page, Canada.ca CRA scam alerts, Canada Post's
phishing guidance) — e.g. the "reply INFO/YES/HELP" call-to-action,
cashback-via-e-Transfer scams, "incomplete address" Canada Post texts,
and CRA "account error, text HELP" scams — not copied verbatim, but
the same structural patterns real scammers use.

random_state is fixed (SEED) so re-running this script always produces
the exact same dataset -- reproducible, like train_test_split's random_state.
"""
import csv
import os
import random

SEED = 42
random.seed(SEED)

PER_CLASS = 250  # target rows per class (3 scam classes + legit = ~1000 total)

BANKS = ["RBC", "TD", "Scotiabank", "CIBC", "BMO", "Tangerine", "Desjardins",
         "National Bank", "HSBC Canada", "Simplii Financial", "Meridian Credit Union",
         "Vancity", "ATB Financial"]
RETAILERS = ["Costco Wholesale Canada", "Walmart Canada", "Canadian Tire", "Loblaws",
             "Amazon.ca", "Best Buy Canada", "Shoppers Drug Mart"]
NAMES = ["John", "Sarah", "Mike", "Priya", "Alex", "Jordan", "Emma", "Liam", "Fatima", "Chris",
         "Dana", "Omar", "Grace", "Kevin", "Noor"]
AMOUNTS = [20, 35, 50, 89.99, 120, 150, 200, 220, 251.76, 275, 300, 325, 340, 375, 415, 450,
           510.25, 540, 600, 680, 720, 750, 864.20, 980.50, 1050, 1200, 1500, 1890]
SMALL_FEES = [2.49, 2.75, 2.99, 3.15, 3.20, 3.50, 3.99, 4.20, 4.55, 4.80, 5.40, 5.99, 6.10, 6.75, 7.25]
HOURS = [1, 2, 3, 6, 12, 24, 48]
DAYS = list(range(1, 29))
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["9am", "10am", "11am", "1pm", "2:30pm", "4pm", "6pm", "7:45am"]
DATES = [f"{m} {d}, {y}" for m in ["March", "April", "May", "June", "September"]
         for d in [5, 8, 14, 21] for y in [2026]]

DOMAINS_ETRANSFER = ["interac-deposit-secure.com", "etransfer-claim23.net", "interac-verify.net",
                     "fund-release-canada.com", "etransfer-urgent-claim.com", "interac-unclaimed.com",
                     "etransfer-manual-sin.net", "interac-retry-deposit.com", "interac-ca-login.net",
                     "etransfer-cashback-claim.com"]
DOMAINS_CRA = ["cra-refund-canada.com", "cra-verify-sin.net", "cra-tax-correction.com",
               "cra-payment-portal.net", "cra-refund-deposit.com", "cra-file-review.net",
               "cra-gst-credit.com", "cra-myaccount-relock.com", "cra-audit-verify.com",
               "cra-netfile-verify.com", "cra-document-review.net"]
DOMAINS_CPOST = ["canadapost-redelivery.com", "cpost-tracking.info", "canadapost-clearance.net",
                 "cpost-redeliver.com", "canadapost-duty-pay.com", "cpost-fee-pay.net",
                 "canadapost-import-tax.com", "cpost-border-hold.net", "cpost-broker-fee.net",
                 "canadapost-address-confirm.com"]
PHONES = ["1-800-555-0199", "437-555-0133", "1-888-555-0177", "1-855-555-0144",
          "647-555-0122", "1-866-555-0188", "1-877-555-0161"]
YEARS = [2023, 2024, 2025]
TRACK_IDS = [f"CP{random.randint(40000, 99999)}" for _ in range(30)]
ORDER_IDS = [f"{random.randint(100,999)}-{random.randint(1000000,9999999)}" for _ in range(30)]
PROJECTS = ["LedgerLite", "Nik-Blog", "Fragments", "CineVision", "scamshield-ca"]
UTILITIES = ["Rogers", "Fido", "Telus", "Bell", "Hydro One", "Enbridge Gas"]
APPT_TYPES = ["dentist", "doctor", "eye", "physio", "hair salon"]
ACTIVITIES = ["coffee", "lunch", "the project meeting", "a study session", "dinner"]
PERSONS = ["Dr. Smith", "Dr. Lee", "Dr. Patel", "the hiring team", "your advisor"]

ETRANSFER_TEMPLATES = [
    "You have received an Interac e-Transfer of ${amt} CAD. Click here to deposit: {domain}",
    "{bank} e-Transfer: ${amt} received. Click to deposit before it is returned to sender: {domain}",
    "{bank}: An e-Transfer of ${amt} is waiting for you. Click to claim within {hrs} hours: {domain}",
    "URGENT: Your Interac e-Transfer of ${amt} will be cancelled in {hrs} hours. Click now to deposit: {domain}",
    "{name} sent you an Interac e-Transfer for ${amt}. Follow this link and answer the security question to deposit: {domain}",
    "Interac Alert: Your e-Transfer of ${amt} will expire in {hrs} hour(s). Claim here: {domain}",
    "{bank} Alert: You have an incoming e-Transfer of ${amt}. Secure deposit link: {domain}",
    "Notification: A ${amt} e-Transfer is on hold. Verify your account to release the funds: {domain}",
    "You've got money! Someone sent you ${amt} via Interac. Deposit now: {domain}",
    "You have 1 unclaimed Interac e-Transfer (${amt}). Expires in {hrs} hours. Claim: {domain}",
    "INTERAC: You Received an e-Transfer. Reply INFO for more details.",
    "{retailer}: Your 2% annual cashback of ${amt} is ready to be withdrawn through e-Transfer. Reply YES to claim.",
    "INTERAC e-Transfer: Remember to deposit your Tax Return of ${amt} from (C.R.A) Canada Revenue Agency by {date}.",
    "{bank} Interac Notice: Someone sent you ${amt}. Reply INFO to view the deposit link.",
]

CRA_TEMPLATES = [
    "CRA Notice: You are eligible for a tax refund of ${amt}. Claim now: {domain}",
    "This is the Canada Revenue Agency. You owe ${amt} in unpaid taxes. Failure to pay within {hrs} hours will result in legal action. Call {phone}.",
    "Your CRA account has been suspended due to suspicious activity. Verify your SIN number here to restore access: {domain}",
    "CRA: A warrant has been issued in your name for unpaid taxes. Resolve this immediately by calling {phone}.",
    "Canada Revenue Agency Notice: Your {year} tax return has an error. Confirm your banking details to avoid penalty: {domain}",
    "CRA Refund Alert: ${amt} has been approved for direct deposit. Confirm your account number here: {domain}",
    "Final notice from the CRA: legal proceedings will begin in {hrs} hours unless you settle ${amt} immediately. Call {phone}.",
    "CRA Important Notice: You qualify for the GST/HST credit of ${amt}. Claim before it expires: {domain}",
    "The CRA has flagged your account for an audit. Verify your identity within {hrs} hours to avoid a freeze: {domain}",
    "Canada Revenue Agency: Your {year} tax file is under review. Submit your SIN and date of birth to avoid suspension: {domain}",
    "CRA Alert: There is an error with your account information. Text back HELP to update your file.",
    "CRA Notification: A new tax document is ready to review. Click the link to view and confirm your details: {domain}",
    "Canada Revenue Agency: Your refund of ${amt} is ready. Deposit via Interac e-Transfer by {date}: {domain}",
]

CPOST_TEMPLATES = [
    "Canada Post: Your package could not be delivered due to an unpaid customs fee of ${fee}. Pay now: {domain}",
    "CANADA POST: We attempted to deliver your parcel but failed. Reschedule delivery and pay ${fee} fee: {domain}",
    "Your Canada Post parcel #{track} is on hold. Pay a small clearance fee of ${fee} to release it: {domain}",
    "Delivery Notice: Canada Post could not deliver your item. Confirm your address and pay ${fee} redelivery fee: {domain}",
    "Canada Post: Your shipment is held at customs. Pay the outstanding duty of ${fee} to proceed: {domain}",
    "Canada Post Alert: Your parcel tracking number {track} requires a ${fee} fee before delivery can continue: {domain}",
    "Notice: Canada Post attempted redelivery twice. Final fee of ${fee} due now to avoid return to sender: {domain}",
    "CANADA POST: Your parcel could not clear customs. Settle the ${fee} balance to avoid your item being discarded: {domain}",
    "Canada Post Tracking #{track}: Your item is held at the border. Settle ${fee} duty within {hrs}h or it will be destroyed: {domain}",
    "Your Canada Post item is being held. Confirm your shipping address and pay the ${fee} handling fee to release it: {domain}",
    "Canada Post: We could not deliver your parcel due to an incomplete address. Confirm your address here to reschedule: {domain}",
    "Canada Post: A small shipping fee of ${fee} is required to release your parcel #{track}. Pay here: {domain}",
]

LEGIT_TEMPLATES = [
    "Hey, are we still meeting for {activity} at {time}?",
    "Your appointment with {person} is confirmed for {date} at {time}.",
    "Reminder: your {utility} bill of ${amt} is due on the {day}.",
    "Your {company} order #{orderid} has shipped and will arrive {day_name}.",
    "Hi, this is a reminder that your {appt_type} appointment is {day_name} at {time}. Reply C to confirm.",
    "Your Interac e-Transfer to {name} for ${amt} was successfully sent and has been deposited.",
    "Thanks for your payment. Your {bank} credit card statement balance is now ${amt}.",
    "Your package from Canada Post was delivered today at {time} to your front door.",
    "Your {bank} account statement for {month} is now available online.",
    "Hey, can you send me the {project} repo link again?",
    "Your CRA Notice of Assessment for the {year} tax year is now available in My Account. No action needed if you agree with it.",
    "Your {utility} bill of ${amt} was successfully charged to your card ending in {last4}.",
    "Hi Nikan, just a heads up that the {project} deployment finished successfully.",
    "Your {company} pickup order will be ready at {time} today.",
    "Reminder: your Seneca assignment for {project} is due {day_name}.",
    "Hey, the {project} demo went great today, nice work.",
    "Your Canada Post tracking number {track} shows your parcel is out for delivery today.",
    "Hi, this is Seneca IT support confirming your password reset request was completed successfully.",
    "Your {bank} Visa payment of ${amt} was received, thank you.",
    "Reminder from your gym: your membership renews automatically on the {day} for ${amt}.",
    "Mom: don't forget {person}'s birthday dinner is this {day_name} at {time}.",
    "Hey it's mom, can you pick up some milk on your way home?",
    "Your Uber is arriving in 3 minutes. Driver: {name}, Toyota Corolla.",
    "Netflix: Your monthly subscription payment was successful. Thanks for being a member.",
    "Your Aurora Marine weekly report has been submitted successfully.",
    "Hey, can we push the mock interview to {day_name} morning instead?",
    "Your Spotify Premium renews on the {day} for $11.99.",
    "Hi, confirming your flight check-in for {date}'s {time} departure is complete.",
    "Your GitHub pull request on {project} was approved and merged.",
    "Hey, loved your explanation of TF-IDF earlier, finally makes sense.",
]


def generate_class(templates, placeholder_fn, count):
    rows = set()
    attempts = 0
    while len(rows) < count and attempts < count * 40:
        attempts += 1
        template = random.choice(templates)
        text = template.format(**placeholder_fn())
        rows.add(text)
    return list(rows)


def scam_placeholders():
    return dict(
        amt=random.choice(AMOUNTS),
        fee=random.choice(SMALL_FEES),
        bank=random.choice(BANKS),
        retailer=random.choice(RETAILERS),
        name=random.choice(NAMES),
        hrs=random.choice(HOURS),
        domain=random.choice(DOMAINS_ETRANSFER + DOMAINS_CRA + DOMAINS_CPOST),
        phone=random.choice(PHONES),
        year=random.choice(YEARS),
        track=random.choice(TRACK_IDS),
        date=random.choice(DATES),
    )


def legit_placeholders():
    return dict(
        activity=random.choice(ACTIVITIES),
        time=random.choice(TIMES),
        person=random.choice(PERSONS),
        date=random.choice(DATES),
        utility=random.choice(UTILITIES),
        day=random.choice(DAYS),
        company=random.choice(RETAILERS),
        orderid=random.choice(ORDER_IDS),
        day_name=random.choice(DAY_NAMES),
        appt_type=random.choice(APPT_TYPES),
        bank=random.choice(BANKS),
        month=random.choice(MONTHS),
        project=random.choice(PROJECTS),
        year=random.choice(YEARS),
        last4=random.randint(1000, 9999),
        amt=random.choice(AMOUNTS),
        name=random.choice(NAMES),
        track=random.choice(TRACK_IDS),
    )


def main():
    rows = []

    for text in generate_class(ETRANSFER_TEMPLATES, scam_placeholders, PER_CLASS):
        rows.append((text, "fake_etransfer"))

    for text in generate_class(CRA_TEMPLATES, scam_placeholders, PER_CLASS):
        rows.append((text, "fake_cra"))

    for text in generate_class(CPOST_TEMPLATES, scam_placeholders, PER_CLASS):
        rows.append((text, "fake_canada_post"))

    for text in generate_class(LEGIT_TEMPLATES, legit_placeholders, PER_CLASS):
        rows.append((text, "legit"))

    random.shuffle(rows)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "scam_dataset.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
